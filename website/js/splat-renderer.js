/* Minimal WebGL2 Gaussian-splat renderer (EWA splatting, front-to-back "under" compositing).
 * Several splat sets can be uploaded and drawn into different viewports/scissors of one canvas
 * with a shared camera, which is what the agent-vs-baseline comparison needs.
 */
const VS = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D uTex;      // RGBA32F, 3 texels per splat, 3072 texels wide
uniform mat4 uView, uProj;
uniform vec2 uFocal;         // pixels, for the current viewport
uniform vec2 uViewport;      // pixels
uniform float uSplatScale;   // 1 = physical size
in uint aIndex;
out vec4 vColor; out vec2 vPos;
void main() {
  int i = int(aIndex);
  ivec2 t0 = ivec2((i % 1024) * 3, i / 1024);
  vec4 A  = texelFetch(uTex, t0, 0);
  vec4 Bc = texelFetch(uTex, t0 + ivec2(1, 0), 0);
  vec4 C  = texelFetch(uTex, t0 + ivec2(2, 0), 0);
  vec4 cam = uView * vec4(A.xyz, 1.0);
  if (cam.z < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 clip = uProj * cam;
  vec2 ndc = clip.xy / clip.w;
  if (any(greaterThan(abs(ndc), vec2(1.4)))) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  mat3 Vrk = mat3(Bc.x, Bc.y, Bc.z,   Bc.y, Bc.w, C.x,   Bc.z, C.x, C.y) * (uSplatScale * uSplatScale);
  float z2 = cam.z * cam.z;
  mat3 J = mat3(uFocal.x / cam.z, 0.0, 0.0,
                0.0, uFocal.y / cam.z, 0.0,
                -(uFocal.x * cam.x) / z2, -(uFocal.y * cam.y) / z2, 0.0);
  mat3 W = mat3(uView);
  mat3 cov2 = J * W * Vrk * transpose(W) * transpose(J);
  float a = cov2[0][0] + 0.3, d = cov2[1][1] + 0.3, b = cov2[0][1];
  float mid = 0.5 * (a + d);
  float rad = length(vec2(0.5 * (a - d), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  vec2 dir = (abs(b) < 1e-7) ? ((a >= d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(b, l1 - a));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x);
  vec2 corner = vec2(((gl_VertexID & 1) == 0) ? -2.0 : 2.0, ((gl_VertexID & 2) == 0) ? -2.0 : 2.0);
  vec2 offPx = corner.x * major + corner.y * minor;
  gl_Position = vec4(ndc + vec2(offPx.x * 2.0 / uViewport.x, -offPx.y * 2.0 / uViewport.y), 0.0, 1.0);
  vPos = corner;
  float p = C.z;
  float r = floor(p / 65536.0);
  float g = floor((p - r * 65536.0) / 256.0);
  float bl = p - r * 65536.0 - g * 256.0;
  vColor = vec4(r / 255.0, g / 255.0, bl / 255.0, A.w);
}`;

const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos;
out vec4 frag;
void main() {
  float A = -dot(vPos, vPos);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  frag = vec4(B * vColor.rgb, B);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class SplatRenderer {
  constructor(canvas, workerUrl) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: true, premultipliedAlpha: true, depth: false, stencil: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.u = {};
    for (const n of ['uTex', 'uView', 'uProj', 'uFocal', 'uViewport', 'uSplatScale']) this.u[n] = gl.getUniformLocation(prog, n);
    this.aIndex = gl.getAttribLocation(prog, 'aIndex');
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
    this.sets = new Map();
    this.worker = new Worker(workerUrl);
    this.worker.onmessage = (e) => this._onSorted(e.data);
    this.onNeedsRedraw = null;
    this.splatScale = 1.0;      // physical size; the viewer no longer exposes a control for this
  }

  /** Upload a decoded AGSP set under an id (replaces any previous set with that id). */
  setSplats(id, decoded) {
    const gl = this.gl;
    this.removeSet(id);
    const n = decoded.count;
    const rows = Math.max(1, Math.ceil(n / 1024));
    const padded = new Float32Array(3072 * rows * 4);
    padded.set(decoded.tex);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 3072, rows, 0, gl.RGBA, gl.FLOAT, padded);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const ibuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
    const seq = new Uint32Array(n); for (let i = 0; i < n; i++) seq[i] = i;
    gl.bufferData(gl.ARRAY_BUFFER, seq, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aIndex);
    gl.vertexAttribIPointer(this.aIndex, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(this.aIndex, 1);
    gl.bindVertexArray(null);
    this.sets.set(id, { tex, vao, ibuf, count: n, sorting: false, pendingView: null, gen: 0, sortedOnce: false });
    const pos = decoded.positions.slice();
    this.worker.postMessage({ type: 'set', id, positions: pos }, [pos.buffer]);
  }

  removeSet(id) {
    const s = this.sets.get(id);
    if (!s) return;
    const gl = this.gl;
    gl.deleteTexture(s.tex); gl.deleteBuffer(s.ibuf); gl.deleteVertexArray(s.vao);
    this.sets.delete(id);
    this.worker.postMessage({ type: 'remove', id });
  }

  has(id) { return this.sets.has(id); }

  _requestSort(id, view) {
    const s = this.sets.get(id);
    if (!s) return;
    if (s.sorting) { s.pendingView = view.slice(); return; }
    s.sorting = true; s.gen++;
    this.worker.postMessage({ type: 'sort', id, view: view.slice(), gen: s.gen });
  }

  _onSorted(m) {
    const s = this.sets.get(m.id);
    if (!s) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, s.ibuf);
    gl.bufferData(gl.ARRAY_BUFFER, m.indices, gl.DYNAMIC_DRAW);
    s.sorting = false; s.sortedOnce = true;
    if (s.pendingView) { const v = s.pendingView; s.pendingView = null; this._requestSort(m.id, v); }
    if (this.onNeedsRedraw) this.onNeedsRedraw();
  }

  /** Resize the drawing buffer to the canvas' CSS size (times a capped device pixel ratio). */
  resize(maxDpr = 1.5) {
    const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; return true; }
    return false;
  }

  /**
   * Draw panels. Each panel: { id, viewport:[x,y,w,h], scissor:[x,y,w,h], view: Float32Array(16), fovY (radians) }.
   * Positions are in drawing-buffer pixels with origin bottom-left.
   */
  render(panels) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.enable(gl.SCISSOR_TEST);
    for (const p of panels) {
      const s = this.sets.get(p.id);
      if (!s) continue;
      const [vx, vy, vw, vh] = p.viewport;
      const [sx, sy, sw, sh] = p.scissor;
      gl.viewport(vx, vy, vw, vh);
      gl.scissor(sx, sy, sw, sh);
      const fy = 0.5 * vh / Math.tan(0.5 * p.fovY);
      const fx = fy;
      const near = 0.05, far = 1000.0;
      const proj = new Float32Array([
        2 * fx / vw, 0, 0, 0,
        0, -2 * fy / vh, 0, 0,
        0, 0, (far + near) / (far - near), 1,
        0, 0, -2 * far * near / (far - near), 0]);
      gl.uniformMatrix4fv(this.u.uView, false, p.view);
      gl.uniformMatrix4fv(this.u.uProj, false, proj);
      gl.uniform2f(this.u.uFocal, fx, fy);
      gl.uniform2f(this.u.uViewport, vw, vh);
      gl.uniform1f(this.u.uSplatScale, this.splatScale);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, s.tex);
      gl.uniform1i(this.u.uTex, 0);
      gl.bindVertexArray(s.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
      gl.bindVertexArray(null);
      this._requestSort(p.id, p.view);
    }
    gl.disable(gl.SCISSOR_TEST);
  }
}
