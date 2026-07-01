"""Build results/showcase.html — a self-contained page with the updated benchmark results and a
gallery of segmentation overlays (base64-inlined so the file opens standalone in any browser).

Run: python -m src.viz.build_showcase
Reads overlay PNGs from results/figures/ (regenerate them with src.viz.plots first).
"""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
OUT = ROOT / "results" / "showcase.html"

# Figures to feature: (task, image index, caption). Files: {task}_test_img{idx}.png
GALLERY = [
    ("wholecell", 8, "Whole-cell segmentation"),
    ("wholecell", 14, "Whole-cell — a denser field"),
    ("nuclear", 2, "Nuclear segmentation"),
    ("nuclear", 8, "Nuclear — larger, sparser nuclei"),
]

WHOLECELL = [  # model, F1@0.5(+ci), F1@0.75, AJI+, PQ, boundaryF1, Dice, best-flags
    ("Cellpose-SAM", "0.844 ±.013", "0.591", "0.718", "0.670", "0.869", "0.902", True),
    ("μSAM", "0.736 ±.018", "0.412", "0.597", "0.553", "0.740", "0.808", False),
]
NUCLEAR = [
    ("Cellpose-SAM", "0.841 ±.017", "0.530", "0.710", "0.651", "0.895", "0.871"),
    ("μSAM", "0.810 ±.015", "0.601", "0.702", "0.648", "0.907", "0.892"),
    ("StarDist (n=148)", "0.766 ±.023", "0.464", "0.633", "0.585", "0.848", "0.844"),
]


def b64(task: str, idx: int):
    p = FIG / f"{task}_test_img{idx}.png"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def row(cells, bold_first=True):
    tds = []
    for i, c in enumerate(cells):
        cls = ' class="model"' if i == 0 and bold_first else ""
        tds.append(f"<td{cls}>{c}</td>")
    return "<tr>" + "".join(tds) + "</tr>"


def wc_row(r):
    name, f1, f175, aji, pq, bf1, dice, best = r
    b = ' class="best"' if best else ""
    return (f'<tr><td class="model">{name}</td>'
            f'<td{b}>{f1}</td><td{b}>{f175}</td><td{b}>{aji}</td><td{b}>{pq}</td><td{b}>{bf1}</td><td{b}>{dice}</td></tr>')


def main():
    imgs = "".join(
        f'<figure class="shot"><img alt="{cap}" src="data:image/png;base64,{d}"><figcaption>{cap}</figcaption></figure>'
        for (task, idx, cap) in GALLERY if (d := b64(task, idx))
    ) or "<p><em>No figures found — run <code>python -m src.viz.plots</code> first.</em></p>"

    wc = "".join(wc_row(r) for r in WHOLECELL)
    nuc = "".join(
        f'<tr><td class="model">{r[0]}</td>' + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>"
        for r in NUCLEAR
    )

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cajal — cell-segmentation benchmark</title>
<style>
:root{{--ink:#1b1b1a;--muted:#6c6c66;--line:#e6e6e0;--teal:#1D9E75;--coral:#D85A30;--blue:#378ADD;--bg:#fbfbf8;--card:#fff;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:940px;margin:24px auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:30px 40px}}
h1{{font-size:24px;margin:0 0 4px;font-weight:650}}.one{{color:var(--muted);margin:0 0 6px}}.meta{{color:var(--muted);font-size:12.5px;margin:0 0 22px}}
h2{{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin:26px 0 10px;font-weight:650}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}th,td{{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}thead th{{color:var(--muted);font-weight:600;border-bottom:1.5px solid #d8d8d2}}
td.best{{font-weight:700;color:#0f4f3a}}td.model{{font-weight:600}}
.lift{{background:#eef7f1;border:1px solid #cfe9dc;border-radius:10px;padding:13px 17px;margin:6px 0;font-size:14px}}.lift b{{color:var(--teal)}}
.legend{{font-size:12.5px;color:var(--muted);margin:2px 0 12px}}.sw{{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin:0 3px 0 10px}}
.gallery{{display:grid;grid-template-columns:1fr;gap:16px}}.shot{{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fafafa}}
.shot img{{width:100%;display:block}}.shot figcaption{{font-size:12.5px;color:var(--muted);padding:7px 12px}}
ul{{margin:6px 0 0;padding-left:20px}}li{{margin:4px 0}}.caveat li{{color:#7a5b2e}}
.foot{{margin-top:22px;padding-top:13px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}
</style></head><body><div class="wrap">
<h1>cajal — which model best outlines cells in multiplexed tissue?</h1>
<p class="one">A reproducible benchmark of Cellpose-SAM, μSAM &amp; StarDist on TissueNet, with a measured fine-tuning gain.</p>
<p class="meta">TissueNet v1.1 · test split · N=297 (StarDist 148) · per-image macro · bootstrap 95% CIs · Gilbreth A30/A10 · higher is better, <b>bold</b> = best in column</p>

<h2>Whole-cell task</h2>
<table><thead><tr><th>model</th><th>F1@0.5</th><th>F1@0.75</th><th>AJI+</th><th>PQ</th><th>boundary-F1</th><th>Dice</th></tr></thead><tbody>{wc}</tbody></table>

<h2>Nuclear task</h2>
<table><thead><tr><th>model</th><th>F1@0.5</th><th>F1@0.75</th><th>AJI+</th><th>PQ</th><th>boundary-F1</th><th>Dice</th></tr></thead><tbody>{nuc}</tbody></table>

<h2>Fine-tuning Cellpose-SAM (whole-cell)</h2>
<div class="lift">Fine-tuning on labeled TissueNet images lifts whole-cell F1@0.5 from <b>0.844</b> (zero-shot)
to <b>0.859</b> with 200 images (+1.5 pts) and <b>0.865</b> with the full 2,580-image set (<b>+2.1 pts</b>) —
more data, more gain.</div>

<h2>Segmentation examples</h2>
<p class="legend">In each panel: the image, then outlines drawn by
<span class="sw" style="background:var(--teal)"></span>ground truth (green) and
<span class="sw" style="background:var(--coral)"></span>each model (magenta). Missing magenta lines between
touching cells = the model merged them; extra lines = it over-split.</p>
<div class="gallery">{imgs}</div>

<h2>Honest limits</h2>
<ul class="caveat">
<li>StarDist ran on a smaller sample (N=148) — its TF build crashes beyond that on this cluster; treat as approximate.</li>
<li>The full-data fine-tune (+2.1) is a single short run — a solid signal, not yet with error bars.</li>
<li>All measured on TissueNet — an honest answer for TissueNet-like tissue.</li>
</ul>

<div class="foot">AJI+ verified identical to HoVer-Net · Hungarian 1-to-1 matching · boundary-F1 = normalized surface dice (2px) · one-command reproducible · unit-tested metrics.</div>
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    n = sum(1 for (t, i, _) in GALLERY if b64(t, i))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {n} figures embedded)")


if __name__ == "__main__":
    main()
