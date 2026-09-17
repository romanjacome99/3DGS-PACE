/* Interactive 3-D viewer: agent vs fixed schedule, same camera, same wall-clock time. */
import { decodeAGSP } from './agsp.js?v=3';
import { SplatRenderer } from './splat-renderer.js?v=3';
import { COLORS, ACTION_LABEL, lineChart, stripChart, legend, fmtK, fmtT, showTip, hideTip } from './charts.js?v=3';
import { tagSeconds, tagLabel, timelineFor } from './tags.js?v=3';

// ---------------------------------------------------------------- tiny vec/mat helpers
const v3 = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
};

/** Column-major world->camera matrix from camera basis columns (r, d, f) and eye. */
function viewMatrix(r, d, f, eye) {
  // W2C rotation rows are r, d, f ; translation = -R^T eye
  const t = [-v3.dot(r, eye), -v3.dot(d, eye), -v3.dot(f, eye)];
  return new Float32Array([
    r[0], d[0], f[0], 0,
    r[1], d[1], f[1], 0,
    r[2], d[2], f[2], 0,
    t[0], t[1], t[2], 1]);
}

export class Viewer {
  constructor(root, manifest, dataBase) {
    this.root = root;
    this.M = manifest;
    this.base = dataBase;
    this.scene = 'train';
    this.backend = '3dgs';
    this.layout = 'side';       // side | wipe
    this.split = 0.5;
    this.tags = ['init'];       // real timeline comes from the scene, in _setScene
    this.tagIndex = 0;          // _setScene lands on the converged model
    this.playing = false;
    this.presetIndex = 0;
    this.photoAlpha = 0;
    this.cache = new Map();     // url -> decoded
    this.cacheOrder = [];
    this.loading = new Map();   // url -> promise
    this.onSelectionChange = null;
    this.atPreset = true;
    this.pointerActive = false;
    this._loadToken = 0;
    this._buildDOM();
    this._initGL();
    this._setScene('train');
    this._bind();
  }

  // ------------------------------------------------------------ DOM
  _buildDOM() {
    const r = this.root;
    r.innerHTML = `
      <div class="v-controls">
        <label>Scene
          <select id="v-scene"></select></label>
        <div class="seg" id="v-backend" role="tablist"></div>
        <div class="seg" id="v-layout">
          <button data-v="side" class="on">side by side</button><button data-v="wipe">wipe</button></div>
        <label>View <select id="v-preset"></select></label>
        <label class="photo-ctl" title="Blend the real photograph of the selected test camera over the rendering">Photo
          <input type="range" id="v-photo" min="0" max="1" step="0.05" value="0"></label>
        <button id="v-reset" class="btn-ghost">reset view</button>
      </div>
      <div class="stage" id="stage">
        <canvas id="gl"></canvas>
        <img id="photo-a" class="photo" alt="" hidden><img id="photo-b" class="photo" alt="" hidden>
        <div class="hud hud-a" id="hud-a"></div>
        <div class="hud hud-b" id="hud-b"></div>
        <div class="divider" id="divider" hidden><div class="knob">⇔</div></div>
        <div class="v-loading" id="v-loading" hidden><div class="bar"><div id="v-loading-bar"></div></div><span id="v-loading-text">loading…</span></div>
        <div class="v-help">drag to orbit · scroll to zoom · right-drag / shift-drag to pan</div>
        <div class="v-nogl" id="v-nogl" hidden></div>
      </div>
      <div class="timeline">
        <button id="v-play" class="btn-play" title="play through the snapshots">▶</button>
        <div class="slider-wrap">
          <input type="range" id="v-time" min="0" max="1" step="1" value="0">
          <div class="ticks" id="v-ticks"></div>
        </div>
        <div class="time-readout" id="v-readout"></div>
      </div>
      <div class="v-below">
        <div class="v-col">
          <h4>What the agent decided, block by block <span class="dim">(hover a block)</span></h4>
          <div id="v-strip"></div>
          <div id="v-strip-legend"></div>
          <h4>Current action <span class="dim" id="v-action-when"></span></h4>
          <div id="v-action" class="action-panel"></div>
        </div>
        <div class="v-col">
          <h4>Gaussian count over wall-clock time</h4>
          <div id="v-chart-n"></div>
          <h4>Held-out PSNR at the snapshots</h4>
          <div id="v-chart-psnr"></div>
        </div>
      </div>`;
    this.$ = (id) => r.querySelector('#' + id);
    const sc = this.$('v-scene');
    for (const [k, s] of Object.entries(this.M.scenes)) {
      const o = document.createElement('option'); o.value = k; o.textContent = `${s.label} · ${s.dataset} (${s.role})`; sc.appendChild(o);
    }
    const bk = this.$('v-backend');
    for (const be of ['3dgs', 'fastergs', 'dash']) {
      const b = document.createElement('button'); b.dataset.v = be; b.textContent = this.M.backend_labels[be];
      b.style.setProperty('--seg-color', COLORS.backend[be]);
      if (be === this.backend) b.classList.add('on');
      bk.appendChild(b);
    }
  }

  _initGL() {
    const canvas = this.$('gl');
    try {
      this.R = new SplatRenderer(canvas, new URL('./sort-worker.js?v=3', import.meta.url));
      this.R.onNeedsRedraw = () => this.requestFrame();
    } catch (e) {
      this.R = null;
      const n = this.$('v-nogl'); n.hidden = false;
      n.textContent = 'WebGL2 is not available in this browser, so the live 3-D view is disabled. The image strip below still shows the same comparison from a fixed camera.';
    }
  }

  // ------------------------------------------------------------ selection
  get sceneData() { return this.M.scenes[this.scene]; }
  get runs() { return this.sceneData.backends[this.backend].methods; }

  _setScene(scene) {
    this.scene = scene;
    this.$('v-scene').value = scene;
    const S = this.sceneData;
    // the timeline is per scene: a run that hits the 30k cap early has no late tags
    this.tags = timelineFor(S);
    this.tagIndex = this.tags.length - 1;      // open on the converged model, not on early training
    const slider = this.$('v-time');
    slider.max = String(this.tags.length - 1);
    slider.value = String(this.tagIndex);
    const ticks = this.$('v-ticks'); ticks.innerHTML = '';
    for (const t of this.tags) { const sp = document.createElement('span'); sp.textContent = tagLabel(t); ticks.appendChild(sp); }
    const ps = this.$('v-preset'); ps.innerHTML = '';
    S.presets.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = p.name; ps.appendChild(o); });
    // world up from the test cameras (OpenCV camera y points down)
    let up = [0, 0, 0];
    for (const p of S.presets) { const R = p.rotation; up = v3.add(up, [-R[0][1], -R[1][1], -R[2][1]]); }
    this.worldUp = v3.norm(up);
    this.center = S.orbit.center;
    this.presetIndex = 0;
    this._applyPreset(0);
    this._refreshRun();
  }

  _setBackend(be) {
    this.backend = be;
    for (const b of this.$('v-backend').children) b.classList.toggle('on', b.dataset.v === be);
    this._refreshRun();
  }

  /** Snapshot entry for a method at the current tag (or nearest earlier/later available). */
  _snapFor(method, tagIdx) {
    const S = this.sceneData;
    const run = this.runs[method];
    const tag = this.tags[tagIdx];
    if (tag === 'init') return { tag: 'init', t: 0, iter: 0, N: S.init.N, psnr: run.init_psnr, ssim: run.init_ssim, file: S.init.file, render: run.render_init, stats: S.init.stats, shared: true };
    let e = run.snapshots.find(s => s.tag === tag);
    if (!e) {
      // a run that reached the iteration cap before this instant has no tag for it; its final
      // model is what it was showing at that wall-clock, so stand `final` in for any late tag
      const fin = run.snapshots.find(s => s.tag === 'final');
      if (fin && tagSeconds(tag) >= fin.t) e = fin;
    }
    if (!e) {
      // nearest by trigger time
      const want = tagSeconds(tag);
      e = run.snapshots.reduce((b, s) => (Math.abs(s.trigger_s - want) < Math.abs(b.trigger_s - want) ? s : b), run.snapshots[0]);
    }
    return e;
  }

  _refreshRun() {
    if (this.onSelectionChange) this.onSelectionChange(this.scene, this.backend);
    this._renderStrip();
    this._renderCharts();
    this._loadCurrent();
  }

  // ------------------------------------------------------------ data loading
  async _fetchDecoded(rel, onProgress) {
    const url = this.base + rel;
    if (this.cache.has(rel)) return this.cache.get(rel);
    if (this.loading.has(rel)) return this.loading.get(rel);
    const p = (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      const total = +res.headers.get('content-length') || 0;
      let buf;
      if (res.body && total) {
        const reader = res.body.getReader();
        const chunks = []; let got = 0;
        for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (onProgress) onProgress(got / total); }
        buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
        buf = buf.buffer;
      } else buf = await res.arrayBuffer();
      const dec = await decodeAGSP(buf);
      this.cache.set(rel, dec); this.cacheOrder.push(rel);
      while (this.cacheOrder.length > 10) { const old = this.cacheOrder.shift(); if (!this.currentFiles || !this.currentFiles.includes(old)) this.cache.delete(old); else this.cacheOrder.push(old); }
      this.loading.delete(rel);
      return dec;
    })();
    this.loading.set(rel, p);
    return p;
  }

  async _loadCurrent() {
    if (!this.R) { this._updateHUD(); return; }
    const a = this._snapFor('agent', this.tagIndex), b = this._snapFor('baseline', this.tagIndex);
    this.currentSnaps = { agent: a, baseline: b };
    this.currentFiles = [a.file, b.file];
    this._updateHUD();
    const token = ++this._loadToken;
    const L = this.$('v-loading'), bar = this.$('v-loading-bar'), txt = this.$('v-loading-text');
    const prog = [0, 0];
    const show = () => { bar.style.width = (50 * (prog[0] + prog[1])).toFixed(0) + '%'; };
    const need = [a.file, b.file].filter(f => !this.cache.has(f));
    if (need.length) { L.hidden = false; txt.textContent = `loading ${need.length === 2 ? 'both snapshots' : 'snapshot'}…`; show(); }
    try {
      const [da, db] = await Promise.all([
        this._fetchDecoded(a.file, (p) => { prog[0] = p; show(); }),
        this._fetchDecoded(b.file, (p) => { prog[1] = p; show(); })]);
      if (token !== this._loadToken) return;
      this.R.setSplats('agent', da);
      this.R.setSplats('baseline', db);
      L.hidden = true;
      this._updateHUD();
      this.requestFrame();
      this._prefetchNeighbors();
    } catch (e) {
      if (token !== this._loadToken) return;
      txt.textContent = 'failed to load: ' + e.message; bar.style.width = '0%';
      console.error(e);
    }
  }

  _prefetchNeighbors() {
    const next = this.tagIndex + 1 < this.tags.length ? this.tagIndex + 1 : null;
    if (next == null) return;
    for (const m of ['agent', 'baseline']) { const s = this._snapFor(m, next); if (!this.cache.has(s.file)) this._fetchDecoded(s.file).catch(() => {}); }
  }

  // ------------------------------------------------------------ camera
  _applyPreset(i) {
    const p = this.sceneData.presets[i];
    if (!p) return;
    this.presetIndex = i;
    this.$('v-preset').value = i;
    const R = p.rotation;   // camera-to-world, rows
    this.cam = {
      r: [R[0][0], R[1][0], R[2][0]], d: [R[0][1], R[1][1], R[2][1]], f: [R[0][2], R[1][2], R[2][2]],
      eye: p.position.slice(), fovY: 2 * Math.atan(p.height / (2 * p.fy)),
    };
    // orbit target: projection of the scene centre on the optical axis
    const dist = Math.max(0.3, v3.dot(v3.sub(this.center, this.cam.eye), this.cam.f));
    this.orbit = { target: v3.add(this.cam.eye, v3.scale(this.cam.f, dist)), dist, up: v3.scale(this.cam.d, -1) };
    this._orbitFromCam();
    this.atPreset = true;
    this._updatePhoto();
    this.requestFrame();
  }

  _orbitFromCam() {
    const o = this.orbit;
    const v = v3.sub(this.cam.eye, o.target);
    o.dist = v3.len(v);
    // basis perpendicular to up
    const up = o.up;
    let u1 = v3.cross(up, Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]); u1 = v3.norm(u1);
    const u2 = v3.cross(up, u1);
    o.u1 = u1; o.u2 = u2;
    const vn = v3.scale(v, 1 / (o.dist || 1));
    o.el = Math.asin(Math.max(-1, Math.min(1, v3.dot(vn, up))));
    o.az = Math.atan2(v3.dot(vn, u2), v3.dot(vn, u1));
  }

  _camFromOrbit() {
    const o = this.orbit;
    const ce = Math.cos(o.el);
    const dir = v3.add(v3.add(v3.scale(o.u1, ce * Math.cos(o.az)), v3.scale(o.u2, ce * Math.sin(o.az))), v3.scale(o.up, Math.sin(o.el)));
    this.cam.eye = v3.add(o.target, v3.scale(dir, o.dist));
    const f = v3.norm(v3.sub(o.target, this.cam.eye));
    let r = v3.cross(f, o.up); if (v3.len(r) < 1e-6) r = v3.cross(f, o.u1);
    r = v3.norm(r);
    const d = v3.cross(f, r);
    this.cam.f = f; this.cam.r = r; this.cam.d = d;
    this.atPreset = false;
    this._updatePhoto();
    this.requestFrame();
  }

  _bind() {
    const $ = this.$;
    $('v-scene').addEventListener('change', (e) => this._setScene(e.target.value));
    $('v-backend').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) this._setBackend(b.dataset.v); });
    $('v-layout').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      this.layout = b.dataset.v;
      for (const x of $('v-layout').children) x.classList.toggle('on', x === b);
      $('divider').hidden = this.layout !== 'wipe';
      this._updatePhoto(); this.requestFrame();
    });
    $('v-preset').addEventListener('change', (e) => this._applyPreset(+e.target.value));
    $('v-reset').addEventListener('click', () => this._applyPreset(this.presetIndex));
    $('v-photo').addEventListener('input', (e) => { this.photoAlpha = +e.target.value; this._updatePhoto(); });
    $('v-time').addEventListener('input', (e) => { this.tagIndex = +e.target.value; this._onTimeChange(); });
    $('v-play').addEventListener('click', () => this._togglePlay());

    // pointer interaction on the canvas
    const stage = $('stage');
    let drag = null;
    stage.addEventListener('contextmenu', (e) => e.preventDefault());
    stage.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.divider')) return;
      drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey, id: e.pointerId };
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener('pointermove', (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      const o = this.orbit;
      if (drag.btn === 2 || drag.shift) {
        const s = o.dist * 0.0015;
        o.target = v3.add(o.target, v3.add(v3.scale(this.cam.r, -dx * s), v3.scale(this.cam.d, -dy * s)));
      } else {
        o.az -= dx * 0.006;
        o.el = Math.max(-1.5, Math.min(1.5, o.el + dy * 0.006));
      }
      this._camFromOrbit();
    });
    const end = (e) => { if (drag && e.pointerId === drag.id) { drag = null; } };
    stage.addEventListener('pointerup', end); stage.addEventListener('pointercancel', end);
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbit.dist *= Math.exp(e.deltaY * 0.0012);
      this.orbit.dist = Math.max(0.05, Math.min(200, this.orbit.dist));
      this._camFromOrbit();
    }, { passive: false });
    // touch pinch
    let pinch = null;
    stage.addEventListener('touchstart', (e) => { if (e.touches.length === 2) pinch = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }, { passive: true });
    stage.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2 && pinch) {
        const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        this.orbit.dist *= pinch / d; pinch = d; this._camFromOrbit(); e.preventDefault();
      }
    }, { passive: false });
    stage.addEventListener('touchend', () => { pinch = null; });

    // wipe divider
    const div = $('divider');
    let dd = false;
    div.addEventListener('pointerdown', (e) => { dd = true; div.setPointerCapture(e.pointerId); e.stopPropagation(); });
    div.addEventListener('pointermove', (e) => { if (!dd) return; const b = stage.getBoundingClientRect(); this.split = Math.max(0.05, Math.min(0.95, (e.clientX - b.left) / b.width)); this.requestFrame(); });
    div.addEventListener('pointerup', () => { dd = false; }); div.addEventListener('pointercancel', () => { dd = false; });

    new ResizeObserver(() => this.requestFrame()).observe(stage);
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowRight') { this.tagIndex = Math.min(this.tags.length - 1, this.tagIndex + 1); $('v-time').value = this.tagIndex; this._onTimeChange(); }
      if (e.key === 'ArrowLeft') { this.tagIndex = Math.max(0, this.tagIndex - 1); $('v-time').value = this.tagIndex; this._onTimeChange(); }
    });
  }

  _onTimeChange() {
    this._updateHUD();
    this._renderAction();
    if (this.stripChart) this.stripChart.setMarker(this.currentTime());
    if (this.nChart) this.nChart.setMarker(this.currentTime());
    if (this.pChart) this.pChart.setMarker(this.currentTime());
    this._loadCurrent();
  }

  /** Wall-clock of the frame on screen. `final` is an iteration cap, not an instant, so the
   *  agent's own finishing time is what the chart markers should point at. */
  currentTime() { const e = this._snapFor('agent', this.tagIndex); return e ? e.t : 0; }

  _togglePlay() {
    this.playing = !this.playing;
    this.$('v-play').textContent = this.playing ? '❚❚' : '▶';
    if (this.playing) {
      if (this.tagIndex >= this.tags.length - 1) this.tagIndex = 0;
      const step = () => {
        if (!this.playing) return;
        this.$('v-time').value = this.tagIndex; this._onTimeChange();
        if (this.tagIndex >= this.tags.length - 1) { this.playing = false; this.$('v-play').textContent = '▶'; return; }
        this.tagIndex++;
        this._playTimer = setTimeout(step, 1800);
      };
      step();
    } else clearTimeout(this._playTimer);
  }

  // ------------------------------------------------------------ rendering
  requestFrame() { if (this._raf) return; this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); }); }

  _panels() {
    const stage = this.$('stage');
    const cw = stage.clientWidth, ch = stage.clientHeight;
    const stacked = this.layout === 'side' && cw < 640;
    const panels = [];
    if (this.layout === 'wipe') {
      panels.push({ id: 'agent', css: [0, 0, cw, ch], sc: [0, 0, cw * this.split, ch] });
      panels.push({ id: 'baseline', css: [0, 0, cw, ch], sc: [cw * this.split, 0, cw * (1 - this.split), ch] });
    } else if (stacked) {
      panels.push({ id: 'agent', css: [0, 0, cw, ch / 2], sc: [0, 0, cw, ch / 2] });
      panels.push({ id: 'baseline', css: [0, ch / 2, cw, ch / 2], sc: [0, ch / 2, cw, ch / 2] });
    } else {
      panels.push({ id: 'agent', css: [0, 0, cw / 2, ch], sc: [0, 0, cw / 2, ch] });
      panels.push({ id: 'baseline', css: [cw / 2, 0, cw / 2, ch], sc: [cw / 2, 0, cw / 2, ch] });
    }
    return { panels, cw, ch, stacked };
  }

  _draw() {
    if (!this.R) return;
    this.R.resize();
    const { panels, ch } = this._panels();
    const view = viewMatrix(this.cam.r, this.cam.d, this.cam.f, this.cam.eye);
    const dpr = this.R.canvas.width / Math.max(1, this.$('stage').clientWidth);
    const gp = panels.map(p => ({
      id: p.id, view,
      fovY: this.cam.fovY,
      viewport: [Math.round(p.css[0] * dpr), Math.round((ch - p.css[1] - p.css[3]) * dpr), Math.round(p.css[2] * dpr), Math.round(p.css[3] * dpr)],
      scissor: [Math.round(p.sc[0] * dpr), Math.round((ch - p.sc[1] - p.sc[3]) * dpr), Math.round(p.sc[2] * dpr), Math.round(p.sc[3] * dpr)],
    }));
    this.R.render(gp);
    // position HUDs and divider
    const hb = this.$('hud-b');
    const stacked = this.layout === 'side' && this.$('stage').clientWidth < 640;
    hb.style.left = this.layout === 'wipe' ? 'auto' : (stacked ? '10px' : 'calc(50% + 10px)');
    hb.style.right = this.layout === 'wipe' ? '10px' : 'auto';
    hb.style.top = stacked ? 'calc(50% + 10px)' : '10px';
    const div = this.$('divider');
    div.style.left = (this.split * 100) + '%';
    this._placePhotos(panels);
  }

  _placePhotos(panels) {
    const S = this.sceneData; const p = S.presets[this.presetIndex];
    const imgs = [this.$('photo-a'), this.$('photo-b')];
    const visible = this.photoAlpha > 0 && this.atPreset && p && p.gt;
    imgs.forEach((img, i) => {
      const pan = panels[i];
      if (!visible || !pan) { img.hidden = true; return; }
      img.hidden = false;
      img.src = this.base + p.gt;
      img.style.opacity = this.photoAlpha;
      const [x, y, w, h] = pan.css;
      const aspect = p.width / p.height;
      const ih = h, iw = h * aspect;
      img.style.top = y + 'px'; img.style.height = ih + 'px'; img.style.width = iw + 'px';
      img.style.left = (x + (w - iw) / 2) + 'px';
      if (this.layout === 'wipe') {
        // clip each photo to its half of the wipe
        const cw = this.$('stage').clientWidth;
        const left = x + (w - iw) / 2;
        const sx0 = i === 0 ? 0 : cw * this.split, sx1 = i === 0 ? cw * this.split : cw;
        img.style.clipPath = `inset(0 ${Math.max(0, left + iw - sx1)}px 0 ${Math.max(0, sx0 - left)}px)`;
      } else img.style.clipPath = 'none';
    });
  }

  _updatePhoto() { const { panels } = this._panels(); this._placePhotos(panels); }

  // ------------------------------------------------------------ HUD / panels
  _updateHUD() {
    const a = this._snapFor('agent', this.tagIndex), b = this._snapFor('baseline', this.tagIndex);
    const be = this.M.backend_labels[this.backend];
    const hud = (e, title, color, isAgent) => {
      const sub = e.stats && e.stats.subsampled ? `<div class="hud-note">viewer shows ${fmtK(e.stats.kept)} of ${fmtK(e.N)} Gaussians</div>` : '';
      return `<div class="hud-title" style="border-color:${color}">${title}</div>
        <div class="hud-row"><b>${fmtT(e.t)}</b> wall-clock · iter ${e.iter.toLocaleString()}</div>
        <div class="hud-row"><b>${fmtK(e.N)}</b> Gaussians</div>
        <div class="hud-row">PSNR <b>${e.psnr.toFixed(2)} dB</b> · SSIM ${e.ssim.toFixed(3)}</div>${sub}`;
    };
    this.$('hud-a').innerHTML = hud(a, `Agent — ${this.M.policies[this.backend].label}`, COLORS.agent, true);
    this.$('hud-b').innerHTML = hud(b, `Fixed schedule — ${be}`, COLORS.base, false);
    const t = this.currentTime();
    this.$('v-readout').innerHTML = t === 0 ? 'initial point cloud (before optimisation)' : `<b>${t} s</b> of training on <b>${be}</b> — same wall-clock budget for both`;
  }

  _blockAt(blocks, t) {
    // block whose interval (t_prev, t] contains t; at t=0 -> none
    if (t <= 0) return null;
    let prev = 0;
    for (const b of blocks) { if (t > prev && t <= b.t + 1e-6) return b; prev = b.t; }
    return blocks[blocks.length - 1];
  }

  _describe(b) {
    const cont = ['densify_threshold_mult', 'prune_opacity_threshold', 'position_lr_mult', 'feature_lr_mult', 'opacity_lr_mult', 'scaling_lr_mult', 'rotation_lr_mult'];
    let s = `<div class="tip-h">block ${b.block} · ends at ${fmtT(b.t)} · iter ${b.iter.toLocaleString()}</div>`;
    s += `<div>${b.block_steps} iters · densify <b>${b.densify_mode}</b> every ${b.densification_interval} · prune <b>${b.prune_mode}</b> · reset <b>${b.opacity_reset}</b></div>`;
    s += `<div class="dim">${cont.map(k => `${ACTION_LABEL[k].replace(' ×', '')} ${k === 'prune_opacity_threshold' ? b[k].toFixed(4) : b[k].toFixed(2) + '×'}`).join(' · ')}</div>`;
    s += `<div>+${fmtK(b.added)} added · −${fmtK(b.pruned)} pruned → <b>${fmtK(b.gaussians)}</b> Gaussians · reward ${b.reward.toFixed(2)}</div>`;
    return s;
  }

  _renderStrip() {
    const blocks = this.runs.agent.blocks;
    const tMax = Math.max(120, blocks[blocks.length - 1].t);
    this.stripChart = stripChart(this.$('v-strip'), {
      rows: [
        { key: 'densify_mode', label: 'densify mode', colorMap: COLORS.densify_mode },
        { key: 'densification_interval', label: 'densify interval', colorMap: COLORS.densification_interval },
        { key: 'prune_mode', label: 'prune mode', colorMap: COLORS.prune_mode },
        { key: 'opacity_reset', label: 'opacity reset', colorMap: COLORS.opacity_reset },
        { key: 'block_steps', label: 'block length', colorMap: COLORS.block_steps },
      ],
      blocks, tMax, marker: this.currentTime(), describe: (b) => this._describe(b), labelWidth: 110,
    });
    const lg = this.$('v-strip-legend'); lg.innerHTML = '';
    legend(lg, COLORS.densify_mode, ['off', 'conservative', 'default', 'aggressive']);
    legend(lg, COLORS.densification_interval, ['50', '100', '200', '400']);
    legend(lg, COLORS.prune_mode, ['off', 'opacity_only', 'opacity_and_size']);
    legend(lg, COLORS.opacity_reset, ['no_reset', 'reset_if_plateau', 'force_reset']);
    this._renderAction();
  }

  _renderAction() {
    const t = this.currentTime();
    const b = this._blockAt(this.runs.agent.blocks, t);
    const box = this.$('v-action');
    if (!b) { box.innerHTML = '<div class="dim">No decision yet: this is the SfM initialisation, before the first block.</div>'; this.$('v-action-when').textContent = ''; return; }
    this.$('v-action-when').textContent = `— block ${b.block}, governing training up to ${fmtT(b.t)}`;
    const chip = (label, val, color) => `<div class="chip" style="--chip:${color || '#888'}"><span>${label}</span><b>${val}</b></div>`;
    const cont = [
      ['densify_threshold_mult', 0.25, 4, 1], ['prune_opacity_threshold', 0.001, 0.02, 0.005], ['position_lr_mult', 0.25, 4, 1],
      ['feature_lr_mult', 0.25, 2, 1], ['opacity_lr_mult', 0.25, 2, 1], ['scaling_lr_mult', 0.25, 2, 1], ['rotation_lr_mult', 0.25, 2, 1]];
    const bars = cont.map(([k, lo, hi, def]) => {
      const log = k !== 'prune_opacity_threshold';
      const f = (v) => log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
      const x = f(b[k]) * 100, xd = f(def) * 100;
      const txt = log ? b[k].toFixed(2) + '×' : b[k].toFixed(4);
      return `<div class="bar-row"><span class="bar-label">${ACTION_LABEL[k].replace(' ×', '')}</span>
        <div class="bar"><div class="bar-def" style="left:${xd}%"></div><div class="bar-fill" style="width:${x}%"></div></div><span class="bar-val">${txt}</span></div>`;
    }).join('');
    box.innerHTML = `<div class="chips">
        ${chip('block', b.block_steps + ' iters', COLORS.block_steps[b.block_steps])}
        ${chip('densify', b.densify_mode, COLORS.densify_mode[b.densify_mode])}
        ${chip('interval', b.densification_interval + ' iters', COLORS.densification_interval[b.densification_interval])}
        ${chip('prune', b.prune_mode, COLORS.prune_mode[b.prune_mode])}
        ${chip('opacity reset', b.opacity_reset, COLORS.opacity_reset[b.opacity_reset])}
      </div>
      <div class="bars">${bars}</div>
      <div class="dim small">Bars span the allowed range of each multiplier; the tick marks the fixed schedule's default. In this block the backend added ${fmtK(b.added)} and pruned ${fmtK(b.pruned)} Gaussians.</div>`;
  }

  _renderCharts() {
    const A = this.runs.agent, B = this.runs.baseline;
    const S = this.sceneData;
    const tMax = 125;
    this.nChart = lineChart(this.$('v-chart-n'), {
      height: 200, xDomain: [0, tMax],
      series: [
        { x: [0, ...A.blocks.map(b => b.t)], y: [S.init.N, ...A.blocks.map(b => b.gaussians)], color: COLORS.agent, label: 'agent' },
        { x: [0, ...B.blocks.map(b => b.t)], y: [S.init.N, ...B.blocks.map(b => b.gaussians)], color: COLORS.base, label: 'fixed schedule' }],
      xLabel: 'training wall-clock (s)', yLabel: 'Gaussians', xFormat: (v) => v + ' s', yFormat: fmtK, marker: this.currentTime(),
    });
    const pa = [{ t: 0, psnr: A.init_psnr }, ...A.snapshots.filter(s => s.tag !== 'final' || !A.snapshots.some(x => x.tag === 't120s'))];
    const pb = [{ t: 0, psnr: B.init_psnr }, ...B.snapshots.filter(s => s.tag !== 'final' || !B.snapshots.some(x => x.tag === 't120s'))];
    this.pChart = lineChart(this.$('v-chart-psnr'), {
      height: 200, xDomain: [0, tMax],
      series: [
        { x: pa.map(s => s.t), y: pa.map(s => s.psnr), color: COLORS.agent, label: 'agent', points: true },
        { x: pb.map(s => s.t), y: pb.map(s => s.psnr), color: COLORS.base, label: 'fixed schedule', points: true }],
      xLabel: 'training wall-clock (s)', yLabel: 'test PSNR (dB)', xFormat: (v) => v + ' s', yFormat: (v) => v.toFixed(1), marker: this.currentTime(),
    });
  }
}

/** Fixed-view image strip (works without WebGL): agent vs fixed schedule at each snapshot time, plus the photo. */
export class FrameStrip {
  constructor(root, manifest, dataBase) {
    this.root = root; this.M = manifest; this.base = dataBase;
  }
  render(scene, backend) {
    const S = this.M.scenes[scene];
    const R = S.backends[backend].methods;
    const be = this.M.backend_labels[backend];
    const tags = timelineFor(S);
    const cell = (method, tag) => {
      const run = R[method];
      let e;
      if (tag === 'init') e = { render: run.render_init, N: S.init.N, psnr: run.init_psnr, t: 0, iter: 0 };
      else {
        e = run.snapshots.find(s => s.tag === tag);
        const fin = run.snapshots.find(s => s.tag === 'final');
        if (!e && fin && tagSeconds(tag) >= fin.t) e = fin;
      }
      if (!e || !e.render) return `<div class="fcell empty"><span class="dim">not reached</span></div>`;
      return `<div class="fcell"><img loading="lazy" src="${this.base + e.render}" alt="${method} at ${tag}">
        <div class="fcap"><b>${e.psnr.toFixed(2)} dB</b> · ${fmtK(e.N)} G · it ${e.iter.toLocaleString()}</div></div>`;
    };
    const gt = R.agent.render_gt ? `<div class="fcell gt"><img loading="lazy" src="${this.base + R.agent.render_gt}" alt="ground truth photo"><div class="fcap">held-out photo</div></div>` : '';
    this.root.innerHTML = `
      <div class="fgrid" style="--cols:${tags.length + 1}">
        <div class="frow-label"></div>${tags.map(t => `<div class="fhead">${tagLabel(t)}</div>`).join('')}<div class="fhead">reference</div>
        <div class="frow-label" style="color:${COLORS.agent}">Agent<br><span class="dim">${this.M.policies[backend].label}</span></div>${tags.map(t => cell('agent', t)).join('')}${gt}
        <div class="frow-label" style="color:${COLORS.base}">Fixed schedule<br><span class="dim">${be}</span></div>${tags.map(t => cell('baseline', t)).join('')}<div class="fcell empty"></div>
      </div>`;
  }
}
