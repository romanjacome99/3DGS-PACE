/* Decisions explorer (per scene, per backend) and cross-backend transfer matrix. */
import { COLORS, ACTION_LABEL, lineChart, stripChart, legend, multiplesChart, fmtK, fmtT } from './charts.js?v=3';

const CONT_FIELDS = [
  { key: 'densify_threshold_mult', label: 'densify gradient threshold', short: 'densify thr.', lo: 0.25, hi: 4, log: true, def: 1 },
  { key: 'prune_opacity_threshold', label: 'prune opacity threshold', short: 'prune thr.', lo: 0.001, hi: 0.02, log: false, def: 0.005 },
  { key: 'position_lr_mult', label: 'position learning rate', short: 'position lr', lo: 0.25, hi: 4, log: true, def: 1 },
  { key: 'feature_lr_mult', label: 'colour / SH learning rate', short: 'SH lr', lo: 0.25, hi: 2, log: true, def: 1 },
  { key: 'opacity_lr_mult', label: 'opacity learning rate', short: 'opacity lr', lo: 0.25, hi: 2, log: true, def: 1 },
  { key: 'scaling_lr_mult', label: 'scale learning rate', short: 'scale lr', lo: 0.25, hi: 2, log: true, def: 1 },
  { key: 'rotation_lr_mult', label: 'rotation learning rate', short: 'rotation lr', lo: 0.25, hi: 2, log: true, def: 1 },
];

function describeBlock(b) {
  const cont = CONT_FIELDS.map(f => `${f.short} ${f.log ? b[f.key].toFixed(2) + '×' : b[f.key].toFixed(4)}`).join(' · ');
  return `<div class="tip-h">block ${b.block} · ends at ${fmtT(b.t)} · iter ${b.iter.toLocaleString()}</div>
    <div>${b.block_steps} iters · densify <b>${b.densify_mode}</b> every ${b.densification_interval} · prune <b>${b.prune_mode}</b> · reset <b>${b.opacity_reset}</b>${b.stop_intent === 'stop' ? ' · <b style="color:#7b3294">wanted to stop</b>' : ''}</div>
    <div class="dim">${cont}</div>
    <div>+${fmtK(b.added)} / −${fmtK(b.pruned)} → <b>${fmtK(b.gaussians)}</b> Gaussians · val PSNR ${b.val_psnr != null ? b.val_psnr.toFixed(2) : '–'} dB</div>`;
}

/** Summary statistics of a decision log, used for the per-backend comparison text. */
export function summarize(blocks) {
  const n = blocks.length;
  const frac = (k, v) => blocks.filter(b => b[k] === v).length / n;
  const mean = (k, log) => { const vs = blocks.map(b => b[k]).filter(v => v != null); const m = log ? Math.exp(vs.reduce((a, v) => a + Math.log(v), 0) / vs.length) : vs.reduce((a, v) => a + v, 0) / vs.length; return m; };
  const wmean = (k) => { let s = 0, w = 0; for (const b of blocks) { s += b[k] * b.block_steps; w += b.block_steps; } return s / w; };
  return {
    blocks: n,
    densifyOff: frac('densify_mode', 'off'), densifyAggr: frac('densify_mode', 'aggressive'), densifyCons: frac('densify_mode', 'conservative'),
    interval: wmean('densification_interval'), pruneOff: frac('prune_mode', 'off'), resetNo: frac('opacity_reset', 'no_reset'), forceReset: frac('opacity_reset', 'force_reset'),
    blockSteps: mean('block_steps'), stopIntent: frac('stop_intent', 'stop'),
    mult: Object.fromEntries(CONT_FIELDS.map(f => [f.key, mean(f.key, f.log)])),
  };
}

export class Decisions {
  constructor(root, data, manifest) {
    this.root = root; this.D = data; this.M = manifest;
    this.scene = 'train'; this.backend = 'all'; this.xLog = true;
    this.native = data.runs.filter(r => r.kind === 'native');
    this._build();
  }

  runsFor(scene) { return ['3dgs', 'fastergs', 'dash'].map(be => this.native.find(r => r.scene === scene && r.backend === be)).filter(Boolean); }

  /** Scenes ordered by their best per-backend speed-up. */
  sceneOrder() {
    const scenes = [...new Set(this.native.map(r => r.scene))];
    const best = (s) => Math.max(...this.runsFor(s).map(r => r.gmean_speedup || 0));
    return scenes.sort((a, b) => best(b) - best(a));
  }

  _overview() {
    const bes = ['3dgs', 'fastergs', 'dash'];
    const rows = this.sceneOrder().map(s => {
      const runs = this.runsFor(s);
      const r0 = runs[0];
      const best = Math.max(...runs.map(r => r.gmean_speedup || 0));
      const cells = bes.map(be => {
        const r = runs.find(x => x.backend === be);
        if (!r) return '<td class="empty">—</td>';
        const g = r.gmean_speedup;
        const fa = r.final.agentic, fb = r.final.baseline;
        const cls = g && g >= 1.5 ? 'hi' : (g && g < 1 ? 'lo' : '');
        return `<td class="ov ${g === best ? 'best' : ''}" data-scene="${s}" data-be="${be}" tabindex="0"><div class="cell-big ${cls}">${g ? g.toFixed(2) + '×' : '–'}</div>
          <div class="cell-small">${fa.test_psnr.toFixed(2)} vs ${fb.test_psnr.toFixed(2)} dB · ${fmtK(fa.N)} vs ${fmtK(fb.N)} G</div></td>`;
      }).join('');
      return `<tr><th><b>${s}</b><br><span class="dim small">${r0.dataset} · ${r0.scene_role}</span></th>${cells}</tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="matrix overview">
      <thead><tr><th>scene ↓ / backend →</th>${bes.map(b => `<th style="color:${COLORS.backend[b]}">${this.M.backend_labels[b]}</th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="dim small">Geometric-mean time-to-target speed-up of the controller over that backend's fixed schedule, per scene (one 30k-iteration rollout each); below: test PSNR and Gaussian count at the end, agent vs fixed. Green ≥ 1.5×, red &lt; 1×. Click a cell to open that rollout's decision log.</p>`;
  }

  _build() {
    const scenes = this.sceneOrder();
    const label = (s) => { const runs = this.runsFor(s); const best = Math.max(...runs.map(r => r.gmean_speedup || 0)); return `${s} · ${runs[0].dataset} (${runs[0].scene_role}) · best ${best.toFixed(2)}×`; };
    this.root.innerHTML = `
      <h3>Speed-up per scene and backend</h3>
      <div id="d-overview">${this._overview()}</div>
      <h3 style="margin-top:26px">Decision logs</h3>
      <div class="v-controls">
        <label>Scene <select id="d-scene">${scenes.map(s => `<option value="${s}">${label(s)}</option>`).join('')}</select></label>
        <div class="seg" id="d-backend"><button data-v="all" class="on">all backends</button>${['3dgs', 'fastergs', 'dash'].map(b => `<button data-v="${b}" style="--seg-color:${COLORS.backend[b]}">${this.M.backend_labels[b]}</button>`).join('')}</div>
        <label class="chk"><input type="checkbox" id="d-log" checked> log time axis</label>
      </div>
      <div id="d-summary" class="d-summary"></div>
      <div id="d-cards"></div>`;
    this.root.querySelector('#d-scene').addEventListener('change', (e) => { this.scene = e.target.value; this.render(); });
    this.root.querySelector('#d-backend').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return; this.backend = b.dataset.v;
      for (const x of e.currentTarget.children) x.classList.toggle('on', x === b); this.render();
    });
    this.root.querySelector('#d-log').addEventListener('change', (e) => { this.xLog = e.target.checked; this.render(); });
    this.scene = scenes[0];
    this.root.querySelectorAll('td.ov').forEach(td => {
      const open = () => {
        this.scene = td.dataset.scene; this.backend = td.dataset.be;
        this.root.querySelector('#d-scene').value = this.scene;
        for (const x of this.root.querySelector('#d-backend').children) x.classList.toggle('on', x.dataset.v === this.backend);
        this.render();
        this.root.querySelector('#d-cards').scrollIntoView({ behavior: 'smooth', block: 'start' });
      };
      td.addEventListener('click', open);
      td.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => this.render(), 200); });
    this.render();
  }

  render() {
    const runs = this.runsFor(this.scene).filter(r => this.backend === 'all' || r.backend === this.backend);
    const cards = this.root.querySelector('#d-cards');
    cards.innerHTML = '';
    this._summary(this.runsFor(this.scene));
    for (const r of runs) cards.appendChild(this.card(r, { compact: this.backend === 'all' }));
  }

  _summary(runs) {
    const box = this.root.querySelector('#d-summary');
    const rows = runs.map(r => {
      const s = summarize(r.agent_blocks);
      const fa = r.final.agentic, fb = r.final.baseline;
      return `<tr>
        <td><span class="sw" style="background:${COLORS.backend[r.backend]}"></span><b>${r.backend_label}</b></td>
        <td>${r.gmean_speedup ? `<b>${r.gmean_speedup.toFixed(2)}×</b>` : '–'}</td>
        <td>${fmtK(fa.N)} <span class="dim">vs ${fmtK(fb.N)}</span></td>
        <td>${fa.test_psnr.toFixed(2)} <span class="dim">vs ${fb.test_psnr.toFixed(2)} dB</span></td>
        <td>${(100 * (1 - s.densifyOff)).toFixed(0)}% <span class="dim">of blocks</span></td>
        <td>${s.interval.toFixed(0)} <span class="dim">iters (fixed: 100)</span></td>
        <td>${(100 * s.resetNo).toFixed(0)}% <span class="dim">no reset</span></td>
        <td>${s.mult.position_lr_mult.toFixed(2)}× / ${s.mult.feature_lr_mult.toFixed(2)}× / ${s.mult.scaling_lr_mult.toFixed(2)}×</td>
        <td>${s.blockSteps.toFixed(0)}</td>
      </tr>`;
    }).join('');
    box.innerHTML = `<div class="table-wrap"><table class="tbl">
      <thead><tr><th>backend</th><th>speed-up<br><span class="dim">gmean over targets</span></th><th>Gaussians at end<br><span class="dim">agent vs fixed</span></th><th>test PSNR at end</th><th>densification on</th><th>densify interval<br><span class="dim">iteration-weighted</span></th><th>opacity resets</th><th>lr multipliers<br><span class="dim">position / colour / scale (geo-mean)</span></th><th>block length<br><span class="dim">mean iters</span></th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="dim small">Statistics of the agent's decision log on <b>${this.scene}</b> (one rollout per backend, the same rollouts behind the paper's tables). "End" is 30k iterations, or the iteration cap the run hit.</p>`;
  }

  /** Card with curves, action strips, multipliers and the time-to-target table of one protocol run. */
  card(r, opts = {}) {
    const c = document.createElement('div');
    c.className = 'card';
    const fa = r.final.agentic, fb = r.final.baseline;
    const title = r.kind === 'transfer' ? `${r.policy_label} → ${r.backend_label} backend <span class="tag">zero-shot transfer</span>` : `${r.backend_label} <span class="dim">controlled by the ${r.policy_label}</span>`;
    c.innerHTML = `
      <div class="card-head" style="border-color:${COLORS.backend[r.backend]}">
        <h3>${title}</h3>
        <div class="card-meta">scene <b>${r.scene}</b> (${r.dataset}, ${r.scene_role}) · ${r.gmean_speedup ? `time-to-target speed-up <b>${r.gmean_speedup.toFixed(2)}×</b> (gmean)` : ''}
          · at the end: agent <b>${fa.test_psnr.toFixed(2)} dB</b> with <b>${fmtK(fa.N)}</b> Gaussians in ${fmtT(fa.t)} (iter ${fa.iter.toLocaleString()}) vs fixed ${fb.test_psnr.toFixed(2)} dB with ${fmtK(fb.N)} in ${fmtT(fb.t)} (iter ${fb.iter.toLocaleString()})</div>
      </div>
      <div class="card-grid">
        <div><h4>Test PSNR vs wall-clock</h4><div class="ch psnr"></div></div>
        <div><h4>Gaussian count vs wall-clock</h4><div class="ch n"></div></div>
      </div>
      <h4>Discrete decisions per block <span class="dim">(hover)</span></h4>
      <div class="ch strips"></div>
      <div class="ch strips-legend"></div>
      <details ${opts.compact ? '' : 'open'}><summary>Continuous multipliers per block (relative to the fixed schedule's default)</summary><div class="ch mult"></div></details>
      <details ${opts.compact ? '' : 'open'}><summary>Time to reach each PSNR target</summary><div class="ch ttt"></div></details>`;
    const xLog = this.xLog;
    const ca = r.curve.filter(x => x.method === 'agentic'), cb = r.curve.filter(x => x.method === 'baseline');
    const tMax = Math.max(...r.curve.map(x => x.t), r.agent_blocks[r.agent_blocks.length - 1].t);
    const targets = r.time_to_target.map(x => x.target_psnr);
    lineChart(c.querySelector('.ch.psnr'), {
      height: 220, xLog, xMin: 1, xDomain: [xLog ? 1 : 0, tMax],
      series: [
        { x: ca.map(x => x.t), y: ca.map(x => x.test_psnr), color: COLORS.agent, label: 'agent', points: true },
        { x: cb.map(x => x.t), y: cb.map(x => x.test_psnr), color: COLORS.base, label: 'fixed schedule', points: true }],
      hlines: targets.map(t => ({ y: t, color: '#c8c8c8' })),
      xLabel: 'training wall-clock', yLabel: 'test PSNR (dB)', xFormat: fmtT, yFormat: (v) => v.toFixed(1),
    });
    lineChart(c.querySelector('.ch.n'), {
      height: 220, xLog, xMin: 1, xDomain: [xLog ? 1 : 0, tMax],
      series: [
        { x: r.agent_blocks.map(b => b.t), y: r.agent_blocks.map(b => b.gaussians), color: COLORS.agent, label: 'agent' },
        { x: r.baseline_blocks.map(b => b.t), y: r.baseline_blocks.map(b => b.gaussians), color: COLORS.base, label: 'fixed schedule' }],
      xLabel: 'training wall-clock', yLabel: 'Gaussians', xFormat: fmtT, yFormat: fmtK,
    });
    stripChart(c.querySelector('.ch.strips'), {
      rows: [
        { key: 'densify_mode', label: 'densify mode', colorMap: COLORS.densify_mode },
        { key: 'densification_interval', label: 'densify interval', colorMap: COLORS.densification_interval },
        { key: 'prune_mode', label: 'prune mode', colorMap: COLORS.prune_mode },
        { key: 'opacity_reset', label: 'opacity reset', colorMap: COLORS.opacity_reset },
        { key: 'block_steps', label: 'block length', colorMap: COLORS.block_steps },
        { key: 'stop_intent', label: 'stop intent', colorMap: COLORS.stop_intent },
      ],
      blocks: r.agent_blocks, tMax, xLog, xMin: 1, describe: describeBlock, labelWidth: 110,
    });
    const lg = c.querySelector('.ch.strips-legend');
    legend(lg, COLORS.densify_mode, ['off', 'conservative', 'default', 'aggressive']);
    legend(lg, COLORS.densification_interval, ['50', '100', '200', '400']);
    legend(lg, COLORS.prune_mode, ['off', 'opacity_only', 'opacity_and_size']);
    legend(lg, COLORS.opacity_reset, ['no_reset', 'reset_if_plateau', 'force_reset']);
    legend(lg, COLORS.block_steps, ['50', '100', '200']);
    legend(lg, COLORS.stop_intent, ['continue', 'stop']);
    const multBox = c.querySelector('.ch.mult');
    const renderMult = () => multiplesChart(multBox, { fields: CONT_FIELDS, blocks: r.agent_blocks, tMax, xLog, xMin: 1, color: COLORS.backend[r.backend] });
    const det = multBox.closest('details');
    if (det.open) renderMult(); else det.addEventListener('toggle', () => { if (det.open && !multBox.children.length) renderMult(); }, { once: false });
    // time-to-target table
    const rows = r.time_to_target.map(x => `<tr><td>${x.target_psnr} dB</td><td>${x.agentic_s != null ? fmtT(x.agentic_s) : '<span class="dim">not reached</span>'}</td><td>${x.baseline_s != null ? fmtT(x.baseline_s) : '<span class="dim">not reached</span>'}</td><td>${x.speedup ? `<b class="${x.speedup >= 1.5 ? 'hi' : ''}">${x.speedup.toFixed(2)}×</b>` : '–'}</td></tr>`).join('');
    c.querySelector('.ch.ttt').innerHTML = `<div class="table-wrap"><table class="tbl small-tbl"><thead><tr><th>target</th><th>agent</th><th>fixed schedule</th><th>speed-up</th></tr></thead><tbody>${rows}</tbody>
      <tfoot><tr><td>geometric mean</td><td></td><td></td><td><b>${r.gmean_speedup ? r.gmean_speedup.toFixed(2) + '×' : '–'}</b></td></tr></tfoot></table></div>`;
    return c;
  }
}

export class Transfer {
  constructor(root, data, manifest, decisions) {
    this.root = root; this.D = data; this.M = manifest; this.dec = decisions;
    this._build();
  }
  _build() {
    const policies = ['3dgs', 'fastergs', 'dash'], backends = ['3dgs', 'fastergs', 'dash', 'legs'];
    const find = (p, b) => this.D.runs.find(r => r.scene === 'train' && r.policy === p && r.backend === b);
    const cellHTML = (r) => {
      if (!r) return '<td class="empty">—</td>';
      const fa = r.final.agentic, fb = r.final.baseline;
      const cls = r.kind === 'native' ? 'native' : 'transfer';
      return `<td class="${cls}" data-id="${r.id}" tabindex="0"><div class="cell-big">${r.gmean_speedup ? r.gmean_speedup.toFixed(2) + '×' : '–'}</div>
        <div class="cell-small">${fa.test_psnr.toFixed(2)} vs ${fb.test_psnr.toFixed(2)} dB<br>${fmtK(fa.N)} vs ${fmtK(fb.N)} G</div>${r.kind === 'native' ? '<div class="cell-tag">native</div>' : ''}</td>`;
    };
    this.root.innerHTML = `
      <div class="table-wrap"><table class="matrix">
        <thead><tr><th>policy trained on ↓ / runs on →</th>${backends.map(b => `<th style="color:${COLORS.backend[b]}">${this.M.backend_labels[b]}</th>`).join('')}</tr></thead>
        <tbody>${policies.map(p => `<tr><th style="color:${COLORS.backend[p]}">${this.M.policies[p].label}</th>${backends.map(b => cellHTML(find(p, b))).join('')}</tr>`).join('')}</tbody>
      </table></div>
      <p class="dim small">Held-out <b>train</b> scene, 30k iterations. Each cell: geometric-mean time-to-target speed-up over the reached PSNR targets; test PSNR and Gaussian count at the end (agent vs that backend's fixed schedule). Click a cell to see the decision log of that rollout. LeGS was never used for training any controller.</p>
      <div id="t-detail"></div>`;
    this.root.querySelectorAll('td[data-id]').forEach(td => {
      const open = () => {
        const r = this.D.runs.find(x => x.id === td.dataset.id);
        const det = this.root.querySelector('#t-detail'); det.innerHTML = '';
        det.appendChild(this.dec.card(r, { compact: true }));
        this.root.querySelectorAll('td[data-id]').forEach(x => x.classList.toggle('sel', x === td));
        det.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      };
      td.addEventListener('click', open);
      td.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    });
  }
}
