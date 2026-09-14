"""Build the static landing page tables from the protocol results (data/decisions.json).
Injects HTML between <!-- TABLES:START --> and <!-- TABLES:END --> markers in website/index.html.
"""
import json
from pathlib import Path
import numpy as np

W = Path(r"C:\Roman\3DGS_PROPOSAL\website")
D = json.loads((W / "data" / "decisions.json").read_text())
LABEL = {"3dgs": "3DGS", "fastergs": "Faster-GS", "dash": "DashGaussian", "legs": "LeGS"}
SCENES = ["train", "barn", "caterpillar", "ignatius", "bicycle", "stump"]
DATASET = {"train": "T&T", "barn": "T&T", "caterpillar": "T&T", "ignatius": "T&T", "bicycle": "Mip-360", "stump": "Mip-360"}


def fk(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{round(n / 1e3)}k"


def cell(r):
    if r is None:
        return "<td>–</td><td>–</td><td>–</td>"
    fa, fb = r["final"]["agentic"], r["final"]["baseline"]
    g = r["gmean_speedup"]
    gc = f'<td class="{"hi" if g and g >= 1.4 else ("lo" if g and g < 1 else "")}">{g:.2f}×</td>' if g else "<td>–</td>"
    pa, pb = fa["test_psnr"], fb["test_psnr"]
    pc = f"<td><b>{pa:.1f}</b>/{pb:.1f}</td>" if pa - pb >= 0.3 else f"<td>{pa:.1f}/{pb:.1f}</td>"
    nc = f"<td><b>{fk(fa['N'])}</b>/{fk(fb['N'])}</td>" if fa["N"] <= 0.8 * fb["N"] else f"<td>{fk(fa['N'])}/{fk(fb['N'])}</td>"
    return gc + pc + nc


native = [r for r in D["runs"] if r["kind"] == "native"]
rows = []
for s in SCENES:
    cells = "".join(cell(next((r for r in native if r["scene"] == s and r["backend"] == be), None)) for be in ["3dgs", "fastergs", "dash"])
    rows.append(f"<tr><th>{s}</th><td class='dim'>{DATASET[s]}</td>{cells}</tr>")
t1 = f"""<div class="table-wrap"><table class="rt">
<thead><tr><th rowspan="2">scene</th><th rowspan="2"></th><th colspan="3">3DGS</th><th colspan="3">Faster-GS</th><th colspan="3">DashGaussian</th></tr>
<tr>{''.join('<th>speed-up</th><th>PSNR<br><span class="dim">PACE/fixed</span></th><th>Gaussians<br><span class="dim">PACE/fixed</span></th>' for _ in range(3))}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""

legs = [r for r in D["runs"] if r["backend"] == "legs"]
order = ["3dgs", "fastergs", "dash"]
legs.sort(key=lambda r: order.index(r["policy"]))
rows2 = []
for r in legs:
    fa, fb = r["final"]["agentic"], r["final"]["baseline"]
    sp = " ".join(f"<td>{x['speedup']:.2f}×</td>" if x.get("speedup") else "<td>–</td>" for x in r["time_to_target"] if x["target_psnr"] in (16, 17, 18, 19, 20))
    rows2.append(f"<tr><th>{LABEL[r['policy']]} policy</th><td>{fk(fa['N'])}/{fk(fb['N'])}</td><td>{fa['test_psnr']:.2f}/{fb['test_psnr']:.2f}</td>{sp}<td class='hi'><b>{r['gmean_speedup']:.2f}×</b></td></tr>")
t2 = f"""<div class="table-wrap"><table class="rt">
<thead><tr><th>run on LeGS</th><th>Gaussians<br><span class="dim">PACE/LeGS</span></th><th>PSNR @30k<br><span class="dim">PACE/LeGS</span></th><th>16 dB</th><th>17 dB</th><th>18 dB</th><th>19 dB</th><th>20 dB</th><th>gmean</th></tr></thead>
<tbody>{''.join(rows2)}</tbody></table></div>"""

html = W / "index.html"
s = html.read_text(encoding="utf-8")
a, b = "<!-- TABLES:START -->", "<!-- TABLES:END -->"
i0, i1 = s.index(a) + len(a), s.index(b)
s = s[:i0] + "\n" + t1 + "\n<p class=\"cap\"><b>Table 1.</b> Zero-shot acceleration on six held-out scenes. Geometric-mean time-to-target speed-up of PACE over each backend's fixed schedule, and test PSNR (dB) and Gaussian count at the end of the run (PACE/fixed). Bold: PACE at least 0.3 dB ahead or at least 20% smaller. Single seed.</p>\n" + t2 + "\n<p class=\"cap\"><b>Table 2.</b> Zero-shot transfer onto LeGS (held-out <i>train</i>): each trained policy run unchanged on the LeGS backend against LeGS's native per-Gaussian controller; speed-up per PSNR target and geometric mean.</p>\n" + s[i1:]
html.write_text(s, encoding="utf-8")
print("landing tables injected")
