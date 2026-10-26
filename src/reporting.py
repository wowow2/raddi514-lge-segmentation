"""Standalone HTML reporting from canonical metadata and saved metrics only."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from src.config import EVAL_LABELS, LABEL_NAMES
from src.experiment_registry import ArtifactRoots
from src.layout import canonical_catalog, read_json
from src.results import ResultsTable


DATASET_PREAMBLE = """
<h2>EMIDEC dataset</h2>
<p>Short-axis late gadolinium enhancement (LGE) cardiac MRI: 100 training cases
(67 pathological, 33 normal) and 50 blind test cases. LGE highlights myocardial
damage after gadolinium contrast administration.</p>
<p>Typical geometry: median 240 × 256 matrix, approximately 1.46 × 1.46 mm
in-plane spacing, median 7 slices (range 4–10), and approximately 10 mm through-plane
spacing. Geometry varies across cases. These descriptive values were retained
from the project dataset preamble; they are not recomputed by this report.</p>
<p>Labels: 0 background, 1 LV cavity, 2 myocardium, 3 scar/infarct, 4 MVO/no-reflow.
Approximate voxel distribution: 97.2%, 1.41%, 1.21%, 0.19%, and 0.02%, respectively.
MVO is present in 40 of the 67 pathological training cases.</p>
<p>Evaluation uses the original five-fold cross-validation splits, stratified by
case characteristics. Results below are out-of-fold validation results, not
blind-test results. Each output retains its own recorded split identity.</p>
"""


def _text(value):
    return escape(str(value if value is not None else "unknown"), quote=True)


def _score(summary):
    value = summary.get("mean")
    if value is None or not math.isfinite(value):
        return "— (n=0)"
    return f'{value:.3f} ± {summary["std"]:.3f} (n={summary["n"]})'


def _table(headers, rows, row_classes=None):
    return ('<div class="scroll"><table><thead><tr>'
            + ''.join(f'<th>{_text(h)}</th>' for h in headers)
            + '</tr></thead><tbody>'
            + ''.join(f'<tr class="{_text(row_classes[i] if row_classes else "")}">'
                      + ''.join(f'<td>{_text(v)}</td>' for v in row) + '</tr>' for i, row in enumerate(rows))
            + '</tbody></table></div>')


def _metrics(table):
    # Exclude background and any non-evaluation rows from all aggregations.
    table = table.where(lambda row: row.get("class_id") in EVAL_LABELS)
    rows = []
    for label in EVAL_LABELS:
        selected = table.filter(class_id=label)
        present = selected.filter(gt_present=True)
        rows.append([LABEL_NAMES[label], _score(selected.overall("dice")),
                     _score(present.overall("dice")), _score(selected.overall("hd95"))])
    rows.append(["Overall (pooled case/class rows)", _score(table.overall("dice")),
                 _score(table.filter(gt_present=True).overall("dice")),
                 _score(table.overall("hd95"))])
    return _table(["Class", "Dice · all cases", "Dice · GT present", "HD95 (mm) · defined only"], rows)


def write_experiment_report(repo, destination):
    """Render every registered output without discovering legacy directories."""
    repo, destination = Path(repo), Path(destination)
    catalog = canonical_catalog(repo)
    if catalog is None:
        raise ValueError("HTML reporting requires canonical experiment metadata")
    roots = ArtifactRoots(repo)
    sections, overview, overview_classes = [], [], []
    for exp in sorted(set(catalog["membership"].values())):
        experiment = read_json(repo / "experiments" / exp / "experiment.json")
        outputs = []
        for name in sorted(key for key, owner in catalog["membership"].items() if owner == exp):
            record = catalog["outputs"][name]
            metrics_path = roots.resolve(record.get("paths", {}).get("metrics"))
            table = None
            if metrics_path is not None and metrics_path.is_file():
                table = ResultsTable.load(metrics_path, run_id=name)
            checkpoint = record.get("inference", {}).get("checkpoint")
            default = name == catalog["defaults"].get(exp)
            output_class = "output default" if default else "output secondary"
            identity = record.get("data", {}).get("splits_sha256")
            summary = (_score(table.where(lambda r: r.get("class_id") in EVAL_LABELS).overall("dice"))
                       if table is not None else "Metrics unavailable")
            overview.append([exp, name + (" · default" if default else ""), checkpoint,
                             len(table.case_set()) if table is not None else "unknown", summary])
            overview_classes.append(output_class)
            details = _table(["Field", "Recorded value"], [
                ["Dataset", record.get("data", {}).get("dataset")],
                ["Stage / source output", f'{record.get("stage")} / {record.get("source_run") or "none recorded"}'],
                ["Checkpoint", checkpoint], ["Split SHA-256", identity],
                ["Probability space", record.get("inference", {}).get("probability_space")],
                ["Postprocessing", json.dumps(record.get("postprocess", {}), sort_keys=True)],
                ["Merge", json.dumps(record.get("merge", {}), sort_keys=True)],
            ])
            body = _metrics(table) if table is not None else '<p class="notice">Metrics unavailable.</p>'
            title = record.get("display_name") or name
            outputs.append(f'<article class="{output_class}"><h3>{_text(title)}'
                           + (' <span class="badge">default</span>' if default else '')
                           + f'</h3><p><code>{_text(name)}</code></p>{details}{body}<details><summary>Full output metadata and artifact references</summary>'
                           + f'<pre>{_text(json.dumps(record, indent=2))}</pre></details></article>')
        sections.append(f'<section class="experiment" id="{_text(exp)}"><h2>{_text(exp)}</h2>'
                        + f'<p>Model: <code>{_text(json.dumps(experiment.get("model", {}), sort_keys=True))}</code></p>'
                        + ''.join(outputs) + '</section>')
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RADDI 514 · EMIDEC experiments</title>
<style>
:root{{color-scheme:light;--ink:#172a38;--muted:#526674;--line:#dbe3e8}}
body{{font:15px/1.6 system-ui,sans-serif;color:var(--ink);background:#f3f6f8;margin:0}}
main{{max-width:1320px;margin:auto;padding:36px 24px}}h1{{font-size:32px;margin-bottom:0}}
h2{{margin-top:32px}}h3{{font-size:18px;overflow-wrap:anywhere}}p{{max-width:100ch}}
header,article{{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}}
.muted{{color:var(--muted)}}.badge{{font-size:12px;background:#dff1e9;padding:4px 8px;border-radius:20px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{background:#edf3f6}}td{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
input[type=search]{{padding:12px;width:min(600px,90%);font:inherit;border:1px solid var(--line);border-radius:8px}}
body:not(.show-all) .secondary{{display:none}}
summary{{cursor:pointer;color:#16675b}}.notice{{color:#8b3c19}}code{{overflow-wrap:anywhere}}
@media print{{body{{background:white}}input{{display:none}}article{{break-inside:avoid}}}}
</style><main><header><p class="muted">RADDI 514 · segmentation experiments</p>
<h1>EMIDEC experiment report</h1><p class="muted">Generated {stamp} · {len(sections)} experiments · {len(overview)} output views</p>
{DATASET_PREAMBLE}
<h2>How to read the metrics</h2>
<p>Values are mean ± population standard deviation, with the number of defined
case/class observations. Dice includes both-empty masks according to the saved
evaluation convention. GT-present Dice includes only rows explicitly marked
as having ground-truth foreground; older reports may lack that information.
HD95 excludes undefined distances. Overall values pool defined case/class rows;
they are not the unweighted average of class means. Outputs are summarized on
their own saved cases; no common-case filtering or split harmonization is applied.
Compare split hashes and sample counts before comparing scores.</p>
<p>Canonical experiment/output JSON supplies identity and provenance. Saved metric
files supply scores. BEST and FINAL outputs remain separate. Unknown fields stay
unknown; historical provenance may mention old paths. No image arrays or plots
are loaded. Plots and per-case exploration are available in the viewer.</p></header>
<h2>Experiment results</h2>
<label><input id="defaults-only" type="checkbox" checked> Show default outputs only</label>
<p class="muted">One default output per experiment is shown initially. Uncheck to include raw, historical and other retained output views.</p>
{_table(["Experiment", "Output", "Checkpoint", "Metric cases", "Overall Dice"], overview, overview_classes)}
<label for="filter">Filter experiment details</label><br><input id="filter" type="search" placeholder="Experiment, output, dataset or checkpoint…">
{''.join(sections)}</main>
<script>document.getElementById('defaults-only').addEventListener('change',function(){{
document.body.classList.toggle('show-all',!this.checked);}});
document.getElementById('filter').addEventListener('input',function(){{
const q=this.value.toLowerCase();document.querySelectorAll('.experiment').forEach(function(s){{
s.hidden=!s.textContent.toLowerCase().includes(q);}});}});</script></html>'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination
