import { Viewer, FrameStrip } from './viewer.js?v=3';
import { Decisions, Transfer } from './decisions.js?v=3';
import { Population } from './population.js?v=3';
import { Ablation } from './ablation.js?v=3';

// Where the data bundle lives. Relative by default (same GitHub Pages site); point this at another
// host (e.g. a release asset folder or a bucket with CORS) if you prefer to keep the ~180 MB of splats out of the repo.
const DATA_BASE = (window.AGS_DATA_BASE || 'data/').replace(/\/?$/, '/');

async function loadJSON(rel) {
  const r = await fetch(DATA_BASE + rel);
  if (!r.ok) throw new Error(`HTTP ${r.status} loading ${rel}`);
  return r.json();
}

function setStatus(msg, isError = false) {
  const s = document.getElementById('boot-status');
  if (!s) return;
  s.textContent = msg; s.hidden = !msg; s.classList.toggle('err', isError);
}

(async () => {
  try {
    setStatus('loading data…');
    const [manifest, decisions] = await Promise.all([loadJSON('manifest.json'), loadJSON('decisions.json')]);
    setStatus('');
    const viewer = new Viewer(document.getElementById('viewer-root'), manifest, DATA_BASE);
    window.agsViewer = viewer;   // handy for debugging in the console
    const strip = new FrameStrip(document.getElementById('frames-root'), manifest, DATA_BASE);
    const pop = new Population(document.getElementById('population-root'), manifest, decisions, DATA_BASE);
    viewer.onSelectionChange = (scene, backend) => {
      strip.render(scene, backend);
      pop.setSelection(scene, backend);
      const s = manifest.scenes[scene];
      document.getElementById('frames-caption').innerHTML = `<b>${s.label}</b> (${s.dataset}) on <b>${manifest.backend_labels[backend]}</b> — the held-out test camera used for the snapshot PSNR; choose the scene and backend in the viewer above.`;
    };
    strip.render(viewer.scene, viewer.backend);
    viewer.onSelectionChange(viewer.scene, viewer.backend);
    const dec = new Decisions(document.getElementById('decisions-root'), decisions, manifest);
    new Transfer(document.getElementById('transfer-root'), decisions, manifest, dec);
    try {
      const abl = await loadJSON('ablation_views.json');
      new Ablation(document.getElementById('ablation-root'), abl);
    } catch (e) { document.getElementById('ablation-root').innerHTML = '<p class="dim">Ablation data not available.</p>'; console.warn(e); }
    // section nav highlighting
    const links = [...document.querySelectorAll('nav a[href^="#"]')];
    const secs = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const io = new IntersectionObserver((es) => {
      for (const e of es) if (e.isIntersecting) { links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }
    }, { rootMargin: '-40% 0px -55% 0px' });
    secs.forEach(s => io.observe(s));
  } catch (e) {
    console.error(e);
    setStatus(`Could not load the data bundle (${e.message}). If you opened index.html from disk, serve the folder over HTTP instead (e.g. "python -m http.server" inside website/).`, true);
  }
})();
