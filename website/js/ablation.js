/* Training-view ablation: does the controller still accelerate with fewer training cameras? */
import { COLORS, lineChart, el, fmtK, fmtT, showTip, hideTip } from './charts.js?v=3';

const SCENE_COLORS = { train: '#0072B2', ignatius: '#E69F00', caterpillar: '#009E73', barn: '#CC79A7' };

/** Small chart with categorical x (view counts) and one line per scene (optionally two styles per scene). */
function viewsChart(container, opts) {
  container.innerHTML = '';
  const W = Math.max(240, container.clientWidth || 400), H = opts.height || 210;
  const m = { l: 52, r: 12, t: 12, b: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = opts.xs;                                  // e.g. [25, 50, 100, 'all']
  const sx = (i) => m.l + (i + 0.5) / xs.length * iw;
  let ys = [];
  for (const s of opts.series) for (const v of s.y) if (v != null) ys.push(v);
  if (opts.yRef != null) ys.push(opts.yRef);
  let y0 = Math.min(...ys), y1 = Math.max(...ys); if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.1; y0 -= pad; y1 += pad;
  if (opts.yMin != null) y0 = Math.min(y0, opts.yMin);
  const sy = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'chart' });
  const nt = 5;
  for (let k = 0; k <= nt; k++) {
    const v = y0 + (y1 - y0) * k / nt;
    svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: 'grid' }));
    svg.appendChild(el('text', { x: m.l - 6, y: sy(v) + 3.5, class: 'tick', 'text-anchor': 'end' }, [opts.yFormat(v)]));
  }
  xs.forEach((x, i) => svg.appendChild(el('text', { x: sx(i), y: m.t + ih + 15, class: 'tick', 'text-anchor': 'middle' }, [String(x)])));
  svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: 'axis' }));
  svg.appendChild(el('text', { x: m.l + iw / 2, y: H - 4, class: 'axis-label', 'text-anchor': 'middle' }, ['training views']));
  if (opts.yRef != null) svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: sy(opts.yRef), y2: sy(opts.yRef), class: 'ref', stroke: '#888' }));
  for (const s of opts.series) {
    let d = '', started = false;
    s.y.forEach((v, i) => { if (v == null) { started = false; return; } d += (started ? 'L' : 'M') + sx(i) + ' ' + sy(v); started = true; });
    svg.appendChild(el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.dash ? 1.5 : 2.2, 'stroke-dasharray': s.dash || null, opacity: s.dash ? 0.7 : 1 }));
    s.y.forEach((v, i) => {
      if (v == null) return;
      const p = s.dash ? el('rect', { x: sx(i) - 3.5, y: sy(v) - 3.5, width: 7, height: 7, fill: s.color, opacity: 0.8 })
                       : el('circle', { cx: sx(i), cy: sy(v), r: 4, fill: s.color, stroke: '#fff', 'stroke-width': 1 });
      p.addEventListener('mousemove', (ev) => showTip(`<b>${s.label}</b> · ${xs[i]} views: ${opts.yFormat(v)}${s.tip ? '<br>' + s.tip[i] : ''}`, ev.clientX, ev.clientY));
      p.addEventListener('mouseleave', hideTip);
      svg.appendChild(p);
    });
  }
  container.appendChild(svg);
}

export class Ablation {
  constructor(root, data) {
    this.root = root; this.D = data;
    this.scenes = Object.keys(data.scenes).filter(s => data.scenes[s].length);
    this.scene = this.scenes[0];
    this.xs = [...data.views, 'all'];
    this._build();
  }

  rowFor(scene, x) {
    const rows = this.D.scenes[scene];
    return x === 'all' ? rows.find(r => r.views > 100) : rows.find(r => r.views === x);
  }

  _build() {
    const done = this.scenes.map(s => `${s}: ${this.D.scenes[s].map(r => r.views > 100 ? 'all' : r.views).join('/')}`).join(' · ');
    this.root.innerHTML = `
      <div class="abl-grid">
        <div><h4>Time-to-target speed-up <span class="dim">(gmean over the PSNR targets both methods reach)</span></h4><div id="a-speed"></div></div>
        <div><h4>Best test PSNR within 30k iterations <span class="dim">(circles agent, squares fixed)</span></h4><div id="a-psnr"></div></div>
        <div><h4>Gaussians at the end <span class="dim">(circles agent, squares fixed)</span></h4><div id="a-n"></div></div>
      </div>
      <div class="cat-legend" id="a-legend"></div>
      <div class="v-controls" style="border-radius:12px;margin-top:18px">
        <label>Scene <select id="a-scene">${this.scenes.map(s => `<option value="${s}">${s}</option>`).join('')}</select></label>
        <span class="dim small">PSNR-vs-time curves for every view count of this scene; the full-view curve is the paper's protocol run.</span>
      </div>
      <div id="a-curves" class="abl-curves"></div>
      <div id="a-table"></div>
      <p class="dim small">Runs available: ${done}. Fewer views also means fewer validation cameras for the controller's state (one fifth of the available views, at most 12); both methods train on the same reduced camera set, chosen by farthest-point sampling with a fixed seed. The SfM initialisation comes from the full capture in all cases.</p>`;
    this.root.querySelector('#a-scene').addEventListener('change', (e) => { this.scene = e.target.value; this._renderScene(); });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => this.render(), 200); });
    this.render();
  }

  render() {
    const xs = this.xs;
    const series = (f, dash, labelSuffix) => this.scenes.map(s => ({
      label: s + (labelSuffix || ''), color: SCENE_COLORS[s] || '#555', dash,
      y: xs.map(x => { const r = this.rowFor(s, x); return r ? f(r) : null; }),
      tip: xs.map(x => { const r = this.rowFor(s, x); return r ? `${r.n_targets} targets · agent ${r.best_psnr.agent.toFixed(2)} dB vs fixed ${r.best_psnr.baseline.toFixed(2)} dB` : ''; }),
    }));
    viewsChart(this.root.querySelector('#a-speed'), { xs, series: series(r => r.gmean_speedup), yRef: 1, yFormat: (v) => v.toFixed(2) + '×' });
    viewsChart(this.root.querySelector('#a-psnr'), { xs, series: [...series(r => r.best_psnr.agent, null, ' agent'), ...series(r => r.best_psnr.baseline, '4 3', ' fixed')], yFormat: (v) => v.toFixed(1) });
    viewsChart(this.root.querySelector('#a-n'), { xs, series: [...series(r => r.final.agent.N, null, ' agent'), ...series(r => r.final.baseline.N, '4 3', ' fixed')], yFormat: fmtK, yMin: 0 });
    const lg = this.root.querySelector('#a-legend'); lg.innerHTML = this.scenes.map(s => `<span><span class="sw" style="background:${SCENE_COLORS[s]}"></span>${s}</span>`).join('');
    this._renderScene();
  }

  _renderScene() {
    const rows = this.D.scenes[this.scene];
    const box = this.root.querySelector('#a-curves'); box.innerHTML = '';
    const cols = Math.min(4, rows.length) || 1;
    box.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    const tMax = Math.max(...rows.flatMap(r => r.curve.map(c => c.t)));
    for (const r of rows) {
      const cell = document.createElement('div');
      cell.innerHTML = `<div class="multiple-title"><b>${r.views > 100 ? 'all ' + r.views : r.views} training views</b> · speed-up ${r.gmean_speedup ? r.gmean_speedup.toFixed(2) + '×' : '–'}</div><div></div>`;
      box.appendChild(cell);
      const a = r.curve.filter(c => c.method === 'agentic'), b = r.curve.filter(c => c.method === 'baseline');
      lineChart(cell.lastElementChild, { height: 190, xLog: true, xMin: 1, xDomain: [1, tMax], margin: { l: 44, r: 8, t: 10, b: 30 },
        series: [{ x: a.map(c => c.t), y: a.map(c => c.psnr), color: COLORS.agent, label: 'agent', points: true },
                 { x: b.map(c => c.t), y: b.map(c => c.psnr), color: COLORS.base, label: 'fixed', points: true }],
        xFormat: fmtT, yFormat: (v) => v.toFixed(0), xLabel: 'wall-clock', yLabel: 'test PSNR (dB)' });
    }
    const tbl = this.root.querySelector('#a-table');
    tbl.innerHTML = `<div class="table-wrap"><table class="tbl small-tbl" style="margin-top:10px"><thead><tr><th>views</th><th>train / val cams</th><th>speed-up (gmean)</th><th>targets reached by both</th><th>best PSNR agent / fixed</th><th>Gaussians at end agent / fixed</th><th>densify on · interval · no-reset</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td><b>${r.views > 100 ? 'all (' + r.views + ')' : r.views}</b></td><td>${r.train_cams ?? '–'} / ${r.val_cams ?? '–'}</td><td>${r.gmean_speedup ? `<b class="${r.gmean_speedup >= 1.5 ? 'hi' : ''}">${r.gmean_speedup.toFixed(2)}×</b>` : '–'}</td><td>${r.n_targets}${r.max_common_target ? ` (up to ${r.max_common_target} dB)` : ''}</td><td>${r.best_psnr.agent.toFixed(2)} / ${r.best_psnr.baseline.toFixed(2)} dB</td><td>${fmtK(r.final.agent.N)} / ${fmtK(r.final.baseline.N)}</td><td>${(100 * r.agent_stats.densify_on_frac).toFixed(0)}% · ${r.agent_stats.interval_mean.toFixed(0)} it · ${(100 * r.agent_stats.no_reset_frac).toFixed(0)}%</td></tr>`).join('')}
      </tbody></table></div>`;
  }
}
