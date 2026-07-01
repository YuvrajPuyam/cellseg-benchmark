"""Build a self-contained, interactive HTML dashboard for the cajal benchmark.

Generates ``results/dashboard.html`` from the artifacts produced by the eval /
fine-tuning / figure stages. The dashboard has three views:

1. Model comparison - the whole-cell and nuclear metric tables, rendered as a
   sortable table plus horizontal metric bars. Parses ``benchmark_tables.md``
   (or the ``*_agg.json`` files if present).
2. Fine-tuning lift - parses ``finetune_delta.md`` (and ``finetune_study.md`` if
   present).
3. Failure-case gallery - every ``results/figures/*.png`` embedded base64-INLINE
   so the HTML is fully standalone, captioned with its task + image index and
   filterable (whole-cell / nuclear).

Stdlib only (matplotlib optional, base64-inlined). No network, no external CSS/JS.

CLI:
  python -m src.viz.build_dashboard
  python -m src.viz.build_dashboard --results results --out results/dashboard.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Palette mirrors results/result_sheet.html for a consistent look.
PALETTE = {
    "ink": "#1b1b1a",
    "muted": "#6c6c66",
    "line": "#e6e6e0",
    "teal": "#1D9E75",
    "coral": "#D85A30",
    "blue": "#378ADD",
    "bg": "#fbfbf8",
    "card": "#fff",
}
# Bar colours cycle per model row.
MODEL_COLORS = ["#1D9E75", "#D85A30", "#378ADD", "#9B6FCB", "#C9A227"]


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _to_num(cell: str):
    """Return a float for a numeric markdown cell, else None. Strips ** bold **."""
    s = cell.strip().strip("*").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_markdown_tables(text: str):
    """Parse a markdown doc into a list of sections.

    Each section is ``{"title": str, "prose": [str], "tables": [table]}`` where
    a table is ``{"header": [str], "rows": [[str]]}``. Section titles come from
    the nearest preceding ``#``/``##``/``###`` heading.
    """
    sections: list[dict] = []
    cur = {"title": "", "prose": [], "tables": []}

    lines = text.splitlines()
    i = 0

    def _flush():
        if cur["title"] or cur["prose"] or cur["tables"]:
            sections.append(cur)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            _flush()
            cur = {"title": heading.group(2).strip(), "prose": [], "tables": []}
            i += 1
            continue
        # Table: a row of pipes followed by a separator row of dashes.
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(
            r"^\|?[\s:|-]+\|?$", lines[i + 1].strip()
        ):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            rows = []
            i += 2  # skip header + separator
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            cur["tables"].append({"header": header, "rows": rows})
            continue
        if stripped and not stripped.startswith("<!--"):
            cur["prose"].append(stripped)
        i += 1

    _flush()
    return sections


def load_agg_json(results: Path):
    """Load ``*_agg.json`` files if present.

    Returns ``{task: {"header": [...], "rows": [[...]]}}`` or ``{}``. Each json is
    expected to map model -> {metric: value}; we render whatever metrics exist.
    """
    aggs = {}
    for jf in sorted(results.glob("*_agg.json")):
        # filename like wholecell_agg.json / nuclear_agg.json
        task = jf.stem.replace("_agg", "")
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict) or not data:
            continue
        # Collect the union of metric keys, preserving first-seen order.
        metrics: list[str] = []
        for mvals in data.values():
            if isinstance(mvals, dict):
                for k in mvals:
                    if k not in metrics:
                        metrics.append(k)
        header = ["model"] + metrics
        rows = []
        for model, mvals in data.items():
            if not isinstance(mvals, dict):
                continue
            row = [str(model)]
            for m in metrics:
                v = mvals.get(m)
                row.append("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
            rows.append(row)
        aggs[task] = {"header": header, "rows": rows}
    return aggs


# --------------------------------------------------------------------------- #
# HTML rendering helpers
# --------------------------------------------------------------------------- #
def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def render_sortable_table(table: dict, table_id: str) -> str:
    """A sortable HTML table. First column = text (left), rest = numeric (right).

    The best (max) value in each numeric column is highlighted.
    """
    header = table["header"]
    rows = table["rows"]
    ncol = len(header)

    # Determine best (max) per numeric column for highlighting.
    best = [None] * ncol
    for c in range(1, ncol):
        vals = [(_to_num(r[c]) if c < len(r) else None) for r in rows]
        nums = [v for v in vals if v is not None]
        best[c] = max(nums) if nums else None

    head_cells = "".join(
        f'<th data-col="{c}" onclick="sortTable(\'{table_id}\',{c})">{esc(h)}'
        f'<span class="arrow"></span></th>'
        for c, h in enumerate(header)
    )

    body = []
    for r in rows:
        tds = []
        for c in range(ncol):
            cell = r[c] if c < len(r) else ""
            num = _to_num(cell)
            cls = []
            if c == 0:
                cls.append("model")
            if best[c] is not None and num is not None and abs(num - best[c]) < 1e-9:
                cls.append("best")
            sort_val = num if num is not None else cell
            cls_attr = f' class="{" ".join(cls)}"' if cls else ""
            tds.append(
                f'<td{cls_attr} data-sort="{esc(sort_val)}">{esc(cell.strip("*"))}</td>'
            )
        body.append("<tr>" + "".join(tds) + "</tr>")

    return (
        f'<table id="{table_id}" class="sortable">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def render_bars(table: dict, metric: str) -> str:
    """Horizontal bars comparing each model on one metric (0..1 scale)."""
    header = [h.lower() for h in table["header"]]
    if "model" not in header or metric.lower() not in header:
        return ""
    mi = header.index("model")
    ci = header.index(metric.lower())
    items = []
    for r in table["rows"]:
        if ci >= len(r):
            continue
        v = _to_num(r[ci])
        if v is None:
            continue
        items.append((r[mi].strip("*"), v))
    if not items:
        return ""
    vmax = max(v for _, v in items) or 1.0
    bars = []
    for k, (name, v) in enumerate(items):
        pct = max(2.0, (v / vmax) * 100.0)
        color = MODEL_COLORS[k % len(MODEL_COLORS)]
        bars.append(
            f'<div class="row"><div class="lab">{esc(name)}</div>'
            f'<div class="track"><div class="bar" style="width:{pct:.1f}%;'
            f'background:{color}"></div></div>'
            f'<div class="val">{v:.3f}</div></div>'
        )
    return (
        f'<div class="bars"><div class="bars-title">{esc(metric)}</div>'
        + "".join(bars)
        + "</div>"
    )


def render_metric_section(title: str, table: dict, idx: int) -> str:
    """One task block: sortable table + bars for a couple of key metrics."""
    tid = f"tbl_{idx}"
    tbl_html = render_sortable_table(table, tid)
    # Choose up to 3 headline metrics that exist in the table.
    preferred = ["f1@0.5", "aji_plus", "aji+", "pq", "boundary_f1", "boundary-f1", "dice"]
    have = [h.lower() for h in table["header"]]
    chosen = []
    for p in preferred:
        if p in have and p not in chosen:
            chosen.append(p)
        if len(chosen) == 3:
            break
    bars = "".join(render_bars(table, table["header"][have.index(m)]) for m in chosen)
    return (
        f'<div class="card metric-card">'
        f"<h3>{esc(title)}</h3>"
        f'<div class="hint">Click a column header to sort.</div>'
        f"{tbl_html}"
        f'<div class="bargroup">{bars}</div>'
        f"</div>"
    )


def render_prose(prose: list[str]) -> str:
    out = []
    for p in prose:
        # Skip pure markdown table separators just in case.
        if re.match(r"^[\s:|-]+$", p):
            continue
        out.append(f"<p>{_inline_md(p)}</p>")
    return "".join(out)


def _inline_md(text: str) -> str:
    """Minimal inline markdown -> HTML: bold, code. Escapes first."""
    s = esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    # bullet leading dash
    s = re.sub(r"^- ", "", s)
    return s


def render_finetune(sections: list[dict]) -> str:
    blocks = []
    for s in sections:
        title = s["title"]
        body = []
        if title:
            body.append(f"<h3>{esc(title)}</h3>")
        if s["prose"]:
            body.append(render_prose(s["prose"]))
        for k, t in enumerate(s["tables"]):
            body.append(render_sortable_table(t, f"ft_{len(blocks)}_{k}"))
            # bars on the lift metric if a delta-ish row exists
            for metric in ("f1@0.5", "aji+", "pq", "boundary-f1", "dice"):
                hdr = [h.lower() for h in t["header"]]
                if metric in hdr:
                    body.append(render_bars(t, t["header"][hdr.index(metric)]))
                    break
        blocks.append(f'<div class="card">{"".join(body)}</div>')
    return "".join(blocks)


def encode_figures(fig_dir: Path):
    """Return list of dicts for each png: name, task, idx, b64."""
    figs = []
    if not fig_dir.is_dir():
        return figs
    for png in sorted(fig_dir.glob("*.png")):
        b = png.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        name = png.stem
        task = "nuclear" if name.startswith("nuclear") else (
            "wholecell" if name.startswith("wholecell") else "other"
        )
        m = re.search(r"img(\d+)", name)
        idx = m.group(1) if m else "?"
        figs.append({"name": name, "task": task, "idx": idx, "b64": b64})
    return figs


def render_gallery(figs: list[dict]) -> str:
    if not figs:
        return '<div class="card"><p class="muted">No figures found in results/figures/.</p></div>'
    cards = []
    for f in figs:
        label = {
            "nuclear": "nuclear",
            "wholecell": "whole-cell",
            "other": "figure",
        }.get(f["task"], f["task"])
        cards.append(
            f'<figure class="gcard" data-task="{esc(f["task"])}">'
            f'<img loading="lazy" alt="{esc(f["name"])}" '
            f'src="data:image/png;base64,{f["b64"]}">'
            f'<figcaption><span class="tag tag-{esc(f["task"])}">{esc(label)}</span>'
            f' image {esc(f["idx"])}</figcaption></figure>'
        )
    return f'<div class="gallery" id="gallery">{"".join(cards)}</div>'


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #
CSS = """
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
header.top{max-width:1100px;margin:0 auto;padding:28px 28px 8px;}
header.top h1{font-size:25px;margin:0 0 4px;font-weight:650;letter-spacing:-.01em;}
header.top .one{color:var(--muted);margin:0 0 6px;font-size:15px;}
header.top .meta{color:var(--muted);font-size:12.5px;margin:0;}
main{max-width:1100px;margin:0 auto;padding:8px 28px 60px;}
nav.tabs{position:sticky;top:0;background:var(--bg);z-index:5;display:flex;gap:6px;
  padding:14px 0 10px;border-bottom:1px solid var(--line);margin-bottom:18px;}
nav.tabs button{font:inherit;font-size:13.5px;border:1px solid var(--line);background:var(--card);
  color:var(--muted);border-radius:20px;padding:6px 16px;cursor:pointer;}
nav.tabs button.active{background:var(--teal);border-color:var(--teal);color:#fff;font-weight:600;}
section.view{display:none;}
section.view.active{display:block;}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:20px 24px;margin:0 0 18px;}
h2.section{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
  margin:6px 0 14px;font-weight:650;}
h3{font-size:16px;margin:0 0 10px;font-weight:640;}
.hint,.muted{color:var(--muted);font-size:12px;}
.hint{margin:0 0 8px;}
table.sortable{width:100%;border-collapse:collapse;font-size:13.5px;margin:0 0 8px;}
table.sortable th,table.sortable td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);}
table.sortable th:first-child,table.sortable td:first-child{text-align:left;}
table.sortable thead th{color:var(--muted);font-weight:600;border-bottom:1.5px solid #d8d8d2;
  cursor:pointer;user-select:none;white-space:nowrap;}
table.sortable thead th:hover{color:var(--ink);}
table.sortable td.best{font-weight:700;color:#0f4f3a;}
table.sortable td.model,.model{font-weight:600;}
.arrow{display:inline-block;width:10px;margin-left:3px;color:var(--teal);font-size:11px;}
.bargroup{display:flex;flex-wrap:wrap;gap:20px;margin-top:14px;}
.bars{flex:1;min-width:280px;}
.bars-title{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
  margin:0 0 6px;font-weight:650;}
.row{display:flex;align-items:center;gap:10px;margin:5px 0;}
.row .lab{width:90px;font-size:12.5px;}
.track{flex:1;background:#f0f0ea;border-radius:5px;height:18px;overflow:hidden;}
.bar{height:18px;border-radius:5px;}
.row .val{width:50px;text-align:right;font-size:12.5px;color:var(--muted);}
p{margin:8px 0;}
code{background:#f2f2ec;border-radius:4px;padding:1px 5px;font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.filterbar{display:flex;gap:8px;margin:0 0 16px;align-items:center;}
.filterbar span{font-size:12px;color:var(--muted);margin-right:4px;}
.filterbar button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);
  color:var(--muted);border-radius:18px;padding:5px 14px;cursor:pointer;}
.filterbar button.active{background:var(--ink);border-color:var(--ink);color:#fff;}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;}
.gcard{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;
  overflow:hidden;}
.gcard img{width:100%;display:block;background:#000;}
.gcard figcaption{padding:8px 12px;font-size:12.5px;color:var(--muted);}
.tag{display:inline-block;border-radius:20px;padding:1px 9px;font-size:11px;font-weight:600;
  margin-right:6px;color:#fff;}
.tag-nuclear{background:var(--blue);}
.tag-wholecell{background:var(--coral);}
.tag-other{background:var(--muted);}
footer{max-width:1100px;margin:0 auto;padding:0 28px 40px;color:var(--muted);font-size:12px;}
"""

JS = """
function showView(id, btn){
  document.querySelectorAll('section.view').forEach(s=>s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}
function sortTable(id, col){
  const t=document.getElementById(id);
  const tb=t.tBodies[0];
  const rows=Array.from(tb.rows);
  const cur=t.getAttribute('data-sort-col');
  const curDir=t.getAttribute('data-sort-dir')||'desc';
  let dir = (String(col)===cur && curDir==='desc') ? 'asc' : 'desc';
  rows.sort((a,b)=>{
    const av=a.cells[col].getAttribute('data-sort');
    const bv=b.cells[col].getAttribute('data-sort');
    const an=parseFloat(av), bn=parseFloat(bv);
    let c;
    if(!isNaN(an)&&!isNaN(bn)) c=an-bn;
    else c=String(av).localeCompare(String(bv));
    return dir==='asc'?c:-c;
  });
  rows.forEach(r=>tb.appendChild(r));
  t.setAttribute('data-sort-col',col);
  t.setAttribute('data-sort-dir',dir);
  t.querySelectorAll('thead th .arrow').forEach(a=>a.textContent='');
  const arrow=t.querySelectorAll('thead th')[col].querySelector('.arrow');
  if(arrow) arrow.textContent = dir==='asc'?'\\u25B2':'\\u25BC';
}
function filterGallery(task, btn){
  document.querySelectorAll('.filterbar button').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.gcard').forEach(c=>{
    c.style.display = (task==='all'||c.dataset.task===task)?'':'none';
  });
}
"""


def build(results: Path, out: Path) -> Path:
    # --- model comparison: prefer agg json, else benchmark_tables.md ---
    aggs = load_agg_json(results)
    metric_blocks = []
    if aggs:
        for k, (task, tbl) in enumerate(sorted(aggs.items())):
            title = {"wholecell": "Whole-cell task", "nuclear": "Nuclear task"}.get(
                task, task
            )
            metric_blocks.append(render_metric_section(title, tbl, k))
        source_note = "source: *_agg.json"
    else:
        bt = results / "benchmark_tables.md"
        if bt.is_file():
            secs = parse_markdown_tables(bt.read_text(encoding="utf-8"))
            k = 0
            for s in secs:
                for t in s["tables"]:
                    raw = s["title"] or f"task {k}"
                    title = {
                        "wholecell task": "Whole-cell task",
                        "nuclear task": "Nuclear task",
                    }.get(raw.lower(), raw[:1].upper() + raw[1:])
                    metric_blocks.append(render_metric_section(title, t, k))
                    k += 1
            source_note = "source: benchmark_tables.md"
        else:
            source_note = "(no benchmark tables found)"
    comparison_html = (
        '<h2 class="section">Model comparison &middot; ' + esc(source_note) + "</h2>"
        + ("".join(metric_blocks) or '<div class="card"><p class="muted">No data.</p></div>')
    )

    # --- fine-tuning ---
    ft_sections = []
    for fname in ("finetune_delta.md", "finetune_study.md"):
        fp = results / fname
        if fp.is_file():
            ft_sections.extend(parse_markdown_tables(fp.read_text(encoding="utf-8")))
    finetune_html = (
        '<h2 class="section">Fine-tuning lift</h2>'
        + (render_finetune(ft_sections) if ft_sections
           else '<div class="card"><p class="muted">No fine-tuning results found.</p></div>')
    )

    # --- gallery ---
    figs = encode_figures(results / "figures")
    n_nuc = sum(1 for f in figs if f["task"] == "nuclear")
    n_wc = sum(1 for f in figs if f["task"] == "wholecell")
    gallery_html = (
        '<h2 class="section">Failure-case gallery &middot; '
        + f"{len(figs)} figures ({n_wc} whole-cell, {n_nuc} nuclear)</h2>"
        '<div class="filterbar"><span>filter:</span>'
        '<button class="active" onclick="filterGallery(\'all\',this)">all</button>'
        '<button onclick="filterGallery(\'wholecell\',this)">whole-cell</button>'
        '<button onclick="filterGallery(\'nuclear\',this)">nuclear</button></div>'
        + render_gallery(figs)
    )

    root_vars = ";".join(f"--{k}:{v}" for k, v in PALETTE.items())
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cajal benchmark dashboard</title>
<style>:root{{{root_vars}}}{CSS}</style>
</head>
<body>
<header class="top">
  <h1>cajal &mdash; cell-segmentation benchmark dashboard</h1>
  <p class="one">Cellpose-SAM vs &micro;SAM on TissueNet v1.1, with a measured fine-tuning lift.</p>
  <p class="meta">test split &middot; per-image (macro) averaging &middot; metrics: AJI+ / PQ / F1@IoU / boundary-F1 / Dice &middot; fully self-contained (images inlined)</p>
</header>
<main>
  <nav class="tabs">
    <button class="active" onclick="showView('view-comparison',this)">Model comparison</button>
    <button onclick="showView('view-finetune',this)">Fine-tuning</button>
    <button onclick="showView('view-gallery',this)">Failure gallery</button>
  </nav>
  <section id="view-comparison" class="view active">{comparison_html}</section>
  <section id="view-finetune" class="view">{finetune_html}</section>
  <section id="view-gallery" class="view">{gallery_html}</section>
</main>
<footer>Generated by <code>src/viz/build_dashboard.py</code> &middot; cajal benchmark &middot; stdlib-only, no network.</footer>
<script>{JS}</script>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the cajal benchmark HTML dashboard.")
    ap.add_argument("--results", default=str(ROOT / "results"),
                    help="results directory (default: <root>/results)")
    ap.add_argument("--out", default=None,
                    help="output html path (default: <results>/dashboard.html)")
    args = ap.parse_args()
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "dashboard.html"
    p = build(results, out)
    size = p.stat().st_size
    print(f"wrote {p} ({size:,} bytes)")


if __name__ == "__main__":
    main()
