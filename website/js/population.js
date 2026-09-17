/* How the Gaussian population evolves along the iterations: count with per-block added/pruned,
 * top-view density thumbnails of the Gaussian centres, and distribution statistics at the snapshots. */
import { COLORS, lineChart, el, fmtK, fmtT, showTip, hideTip } from './charts.js?v=3';
import { tagSeconds, tagLabel, timelineFor } from './tags.js?v=3';

/** Bars of added (up) and pruned (down) Gaussians per block, with the count as a line on a second axis. */
function growthChart(container, opts) {
  container.innerHTML = '';
  const W = Math.max(280, container.clientWidth || 600), H = opts.height || 230;
  const m = { l: 56, r: 56, t: 14, b: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const A = opts.agent, B = opts.baseline, xk = opts.xKey;
  const xmax = Math.max(A[A.length - 1][xk], B[B.length - 1][xk]);
  const sx = (v) => m.l + v / xmax * iw;
  const maxN = Math.max(...A.map(b => b.gaussians), ...B.map(b => b.gaussians), opts.initN);
  const maxAdd = Math.max(1, ...A.map(b => b.added), ...B.map(b => b.added));
  const maxPr = Math.max(1, ...A.map(b => b.pruned), ...B.map(b => b.pruned));
  const barMax = Math.max(maxAdd, maxPr);
  const mid = m.t + ih * 0.62;                 // zero line of the bars
  const syBar = (v) => mid - v / barMax * (ih * 0.62 - 4);
  const syBarNeg = (v) => mid + v / barMax * (ih * 0.38 - 4);
  const syN = (v) => m.t + (1 - v / maxN) * ih;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'chart' });
  // x ticks
  const step = xmax > 20000 ? 5000 : xmax > 8000 ? 2000 : xmax > 3000 ? 1000 : xmax > 150 ? 20 : 500;
  for (let v = 0; v <= xmax; v += step) {
    svg.appendChild(el('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ih, class: 'grid' }));
    svg.appendChild(el('text', { x: sx(v), y: m.t + ih + 15, class: 'tick', 'text-anchor': 'middle' }, [opts.xFormat(v)]));
  }
  svg.appendChild(el('line', { x1: m.l, x2: m.l + iw, y1: mid, y2: mid, class: 'axis' }));
  svg.appendChild(el('text', { x: m.l - 6, y: syBar(barMax) + 4, class: 'tick', 'text-anchor': 'end' }, ['+' + fmtK(barMax)]));
  svg.appendChild(el('text', { x: m.l - 6, y: mid + 4, class: 'tick', 'text-anchor': 'end' }, ['0']));
  svg.appendChild(el('text', { x: m.l - 6, y: syBarNeg(barMax) + 4, class: 'tick', 'text-anchor': 'end' }, ['−' + fmtK(barMax)]));
  svg.appendChild(el('text', { x: 12, y: m.t + ih / 2, class: 'axis-label', 'text-anchor': 'middle', transform: `rotate(-90 12 ${m.t + ih / 2})` }, ['added / pruned per block']));
  svg.appendChild(el('text', { x: W - 8, y: m.t + ih / 2, class: 'axis-label', 'text-anchor': 'middle', transform: `rotate(90 ${W - 8} ${m.t + ih / 2})` }, ['Gaussians (line)']));
  for (const v of [0.25, 0.5, 0.75, 1]) svg.appendChild(el('text', { x: m.l + iw + 6, y: syN(v * maxN) + 4, class: 'tick' }, [fmtK(v * maxN)]));
  svg.appendChild(el('text', { x: m.l + iw / 2, y: H - 4, class: 'axis-label', 'text-anchor': 'middle' }, [opts.xLabel]));
  const drawBars = (blocks, color, shift) => {
    let prev = 0;
    for (const b of blocks) {
      const x0 = sx(prev), x1 = sx(b[xk]); prev = b[xk];
      const w = Math.max(1, (x1 - x0) * 0.46);
      const x = x0 + (shift ? w : 0);
      if (b.added > 0) svg.appendChild(el('rect', { x, y: syBar(b.added), width: w, height: mid - syBar(b.added), fill: color, opacity: 0.75 }));
      if (b.pruned > 0) svg.appendChild(el('rect', { x, y: mid, width: w, height: syBarNeg(b.pruned) - mid, fill: color, opacity: 0.35 }));
      const hit = el('rect', { x: x0, y: m.t, width: Math.max(1, x1 - x0), height: ih, fill: 'transparent' });
      hit.addEventListener('mousemove', (ev) => showTip(`<div class="tip-h">${opts.label(b)}</div>+${fmtK(b.added)} added · −${fmtK(b.pruned)} pruned → <b>${fmtK(b.gaussians)}</b> Gaussians${b.densify_mode ? `<br><span class="dim">densify ${b.densify_mode} every ${b.densification_interval} · prune ${b.prune_mode} · reset ${b.opacity_reset}</span>` : ''}`, ev.clientX, ev.clientY));
      hit.addEventListener('mouseleave', hideTip);
      svg.appendChild(hit);
    }
  };
  drawBars(B, COLORS.base, true);
  drawBars(A, COLORS.agent, false);
  const line = (blocks, color, dash) => {
    let d = `M${sx(0)} ${syN(opts.initN)}`;
    for (const b of blocks) d += `L${sx(b[xk]).toFixed(1)} ${syN(b.gaussians).toFixed(1)}`;
    svg.appendChild(el('path', { d, fill: 'none', stroke: color, 'stroke-width': 2.2, 'stroke-dasharray': dash || null }));
  };
  line(B, COLORS.base); line(A, COLORS.agent);
  // legend
  const lg = [[COLORS.agent, 'agent'], [COLORS.base, 'fixed schedule']];
  let lx = m.l + 8;
  for (const [c, t] of lg) {
    svg.appendChild(el('rect', { x: lx, y: m.t + 2, width: 12, height: 12, fill: c }));
    svg.appendChild(el('text', { x: lx + 16, y: m.t + 12, class: 'legend' }, [t])); lx += 16 + t.length * 6.5 + 14;
  }
  svg.appendChild(el('text', { x: lx + 4, y: m.t + 12, class: 'legend dim' }, ['bars: +added above the axis, −pruned below (lighter); line: total count']));
  container.appendChild(svg);
}

export class Population {
  constructor(root, manifest, decisions, dataBase) {
    this.root = root; this.M = manifest; this.D = decisions; this.base = dataBase;
    this.horizon = 'snap';   // snap (the viewer's own snapshot rollout) | full (the evaluation rollout)
    this.root.innerHTML = `
      <div class="v-controls">
        <span id="p-caption" class="dim"></span>
        <div class="seg" id="p-horizon"><button data-v="snap" class="on">snapshot rollout (drives the viewer)</button><button data-v="full">evaluation rollout</button></div>
      </div>
      <div class="pop-body">
        <h4>Gaussian centres seen from above, coloured by opacity-weighted density <span class="dim">(same frame for every panel)</span></h4>
        <div id="p-clouds"></div>
        <h4>Count, and how many Gaussians each block added or pruned</h4>
        <div id="p-growth"></div>
        <h4>What the population looks like at each snapshot</h4>
        <div id="p-stats" class="pop-stats"></div>
      </div>`;
    this.root.querySelector('#p-horizon').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return; this.horizon = b.dataset.v;
      for (const x of e.currentTarget.children) x.classList.toggle('on', x === b); this.render();
    });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => this.render(), 200); });
  }

  setSelection(scene, backend) { this.scene = scene; this.backend = backend; this.render(); }

  render() {
    if (!this.scene) return;
    const S = this.M.scenes[this.scene], R = S.backends[this.backend].methods, be = this.M.backend_labels[this.backend];
    this.root.querySelector('#p-caption').innerHTML = `<b>${S.label}</b> on <b>${be}</b> — follows the viewer's selection`;
    // thumbnails
    const cl = this.root.querySelector('#p-clouds');
    const cell = (method, tag) => {
      const run = R[method];
      let e = tag === 'init' ? { cloud: S.init.cloud, N: S.init.N, iter: 0 } : run.snapshots.find(s => s.tag === tag);
      if (!e && tag !== 'init') {
        // a run that hit the iteration cap before this instant stayed on its final model
        const fin = run.snapshots.find(s => s.tag === 'final');
        if (fin && tagSeconds(tag) >= fin.t) e = fin;
      }
      if (!e || !e.cloud) return '<div class="fcell empty"><span class="dim">not reached</span></div>';
      return `<div class="fcell cloud"><img loading="lazy" src="${this.base + e.cloud}" alt="${method} Gaussian centres at ${tag}"><div class="fcap"><b>${fmtK(e.N)}</b> Gaussians · it ${e.iter.toLocaleString()}</div></div>`;
    };
    const tags = timelineFor(S);
    cl.innerHTML = `<div class="fgrid" style="--cols:${tags.length}">
      <div class="frow-label"></div>${tags.map(t => `<div class="fhead">${tagLabel(t)}</div>`).join('')}
      <div class="frow-label" style="color:${COLORS.agent}">Agent</div>${tags.map(t => cell('agent', t)).join('')}
      <div class="frow-label" style="color:${COLORS.base}">Fixed schedule</div>${tags.map(t => cell('baseline', t)).join('')}</div>`;
    // growth chart
    const g = this.root.querySelector('#p-growth');
    if (this.horizon === 'snap') {
      growthChart(g, { agent: R.agent.blocks, baseline: R.baseline.blocks, initN: S.init.N, xKey: 'iter', xLabel: 'iteration (the agent picks longer blocks and runs more iterations per second when the model is small)',
        xFormat: (v) => v.toLocaleString(), label: (b) => `block ${b.block} · iter ${b.iter.toLocaleString()} · ${fmtT(b.t)}` });
    } else {
      const run = this.D.runs.find(r => r.kind === 'native' && r.scene === this.scene && r.backend === this.backend);
      if (!run) g.innerHTML = '<p class="dim">No 30k rollout for this scene/backend.</p>';
      else growthChart(g, { agent: run.agent_blocks, baseline: run.baseline_blocks, initN: run.agent_blocks[0].gaussians - run.agent_blocks[0].added + run.agent_blocks[0].pruned, xKey: 'iter',
        xLabel: 'iteration (full evaluation rollout)', xFormat: (v) => fmtK(v), label: (b) => `block ${b.block} · iter ${b.iter.toLocaleString()} · ${fmtT(b.t)}` });
    }
    // stats
    const st = this.root.querySelector('#p-stats'); st.innerHTML = '';
    const series = (method, f) => {
      const run = R[method];
      // `final` repeats the last timed snapshot when the run hit the cap at that instant
      const pts = [{ iter: 0, t: 0, stats: S.init.stats },
                   ...run.snapshots.filter(s => s.tag !== 'final' || !run.snapshots.some(x => x.tag !== 'final' && +x.iter === +s.iter))];
      return { x: pts.map(p => p.iter), y: pts.map(p => p.stats && p.stats.pop ? f(p.stats.pop) : null) };
    };
    const panels = [
      { title: 'median opacity', f: (p) => p.opacity_q[1], fmt: (v) => v.toFixed(2) },
      { title: 'near-transparent fraction (opacity < 0.01)', f: (p) => p.near_transparent_frac, fmt: (v) => (100 * v).toFixed(1) + '%' },
      { title: 'median splat radius (scene units)', f: (p) => p.scale_q[1], fmt: (v) => v.toExponential(1) },
      { title: 'median anisotropy (longest / shortest axis)', f: (p) => p.aniso_q[0], fmt: (v) => v.toFixed(1) },
    ];
    for (const pn of panels) {
      const box = document.createElement('div'); box.className = 'multiple';
      box.innerHTML = `<div class="multiple-title">${pn.title}</div><div></div>`;
      st.appendChild(box);
      const a = series('agent', pn.f), b = series('baseline', pn.f);
      lineChart(box.lastElementChild, { height: 156, margin: { l: 46, r: 8, t: 8, b: 32 },
        series: [{ x: a.x, y: a.y, color: COLORS.agent, label: 'agent', points: true }, { x: b.x, y: b.y, color: COLORS.base, label: 'fixed', points: true }],
        xLabel: 'iteration', xFormat: (v) => fmtK(v), yFormat: pn.fmt, legend: false });
    }
  }
}
