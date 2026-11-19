"""Render a standalone HTML report from the triangle attention replay summary."""

from collections import defaultdict
from html import escape

from accuracy_gate import CANDIDATE, CAPTURE, METRICS


STYLES = """
  * { box-sizing: border-box; }
  body { max-width: 76rem; margin: 2rem auto; padding: 0 1rem;
    font: 1rem/1.5 system-ui, sans-serif; color: #16283a; background: #f5f7fa; }
  h1 { margin-bottom: .25rem; }
  h2 { font-size: 1.15rem; margin: 0 0 .5rem; }
  a { color: #1355a0; }
  .status { font-weight: 700; }
  .passed { color: #166343; }
  .failed { color: #a52831; }
  .group { margin: 1rem 0; padding: 1rem; border: 1px solid #d3dce7;
    background: #fff; border-radius: .5rem; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { border-top: 1px solid #e1e7ef; padding: .5rem; text-align: right; }
  th:first-child { text-align: left; }
  .chart { width: 30%; min-width: 9rem; padding-right: 1rem; }
  .plot { position: relative; display: grid; gap: .2rem; background: #edf1f6; }
  .bar { height: .5rem; }
  .capture { background: #526b89; }
  .candidate { background: #16766d; }
  .limit { position: absolute; inset-block: -.15rem; border-left: 2px dashed #18283c; }
  .legend { font-weight: 650; }
  .capture-label { color: #405875; }
  .candidate-label { color: #11645d; }
"""


def format_metric(value):
  return '—' if value is None else f'{value:.6g}'


def metric_row(metric, summaries, gate_group):
  capture = summaries.get(CAPTURE, {}).get(f'worst_{metric}')
  candidate = summaries.get(CANDIDATE, {}).get(f'worst_{metric}')
  stats = gate_group.get('metrics', {}).get(metric, {})
  threshold = gate_group.get('thresholds', {}).get(metric)
  capture_max = capture[metric] if capture else None
  candidate_max = candidate[metric] if candidate else None
  scale = max(capture_max or 0, candidate_max or 0, threshold or 0) or 1
  marker = '' if threshold is None else (
      f'<span class="limit" style="left:{threshold / scale:.4%}"></span>'
  )
  return f'''<tr>
    <th scope="row"><code>{metric}</code></th>
    <td class="chart"><div class="plot" aria-hidden="true">
      <div class="bar capture" style="width:{(capture_max or 0) / scale:.4%}"></div>
      <div class="bar candidate" style="width:{(candidate_max or 0) / scale:.4%}"></div>
      {marker}</div></td>
    <td>{format_metric(capture_max)}</td><td>{format_metric(candidate_max)}</td>
    <td>{format_metric(threshold)}</td><td>{format_metric(stats.get('median'))}</td>
    <td>{format_metric(stats.get('p95'))}</td></tr>'''


def write_report(path, report):
  gate = report['accuracy_gate']
  group_fields = ('input', 'direction', 'mask_group')
  summaries = defaultdict(dict)
  for group in report['groups']:
    key = tuple(group[field] for field in group_fields)
    summaries[key][group['comparison']] = group
  gate_groups = {
      tuple(group[field] for field in group_fields): group
      for group in gate.get('groups', [])
  }

  sections = []
  for key, comparisons in sorted(summaries.items()):
    gate_group = gate_groups.get(key, {})
    heading = escape(' / '.join(key))
    status = ''
    if gate_group:
      violations = len(gate_group['violations'])
      status_class = 'passed' if gate_group['passed'] else 'failed'
      status = f' <span class="status {status_class}">· {violations} violations</span>'
    rows = '\n'.join(metric_row(metric, comparisons, gate_group) for metric in METRICS)
    sections.append(f'''<section class="group">
      <h2>{heading}{status}</h2>
      <table aria-label="{heading}">
        <thead><tr><th scope="col">Metric</th><th scope="col">Comparison</th>
          <th scope="col">Capture max</th><th scope="col">Candidate max</th>
          <th scope="col">Threshold</th><th scope="col">Candidate median</th>
          <th scope="col">Candidate P95</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></section>''')

  reasons = ''.join(f'<li>{escape(reason)}</li>' for reason in gate.get('reasons', []))
  reasons = f'<ul>{reasons}</ul>' if reasons else ''
  charts = '\n'.join(sections) or '<p>No error metrics were recorded.</p>'
  quality_violations = report['quality_gate']['violation_count']
  status = escape(gate['status'])
  threshold_violations = gate.get('violation_count')
  gate_count = '' if threshold_violations is None else (
      f' · {threshold_violations:,} gate threshold violations'
  )
  nonfinite = sum(group['nonfinite_count'] for group in report['groups'])
  path.write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Triangle attention accuracy report</title><style>{STYLES}</style></head>
<body><main>
<h1>Triangle attention accuracy report</h1>
<p><code>{escape(report['candidate']['name'])}</code> · {escape(gate['scope'])} scope ·
  Accuracy gate <span class="status {status}">{status}</span>{gate_count}</p>
<p>{report['event_count']:,} selected events · {report['i_count']:,} selected i values ·
  {report['metric_rows']:,} metric rows · {len(report['failures']):,} failures ·
  {nonfinite:,} nonfinite values · {quality_violations:,} manual tolerance violations</p>
{reasons}
<p><a href="errors.csv">Per-i/head errors (CSV)</a> ·
  <a href="run_summary.json">Run summary (JSON)</a></p>
<p>Errors are measured against the NumPy FP64 reference. Each chart uses its own
  linear scale from zero. The <span class="legend capture-label">top bar</span> shows
  the capture maximum, the <span class="legend candidate-label">bottom bar</span> shows
  the candidate maximum, and the dashed line marks the gate threshold. An em dash
  indicates an unavailable value. Relative L2 is shown as a ratio.</p>
{charts}
</main></body></html>
''', encoding='utf-8')
