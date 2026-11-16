"""Groupwise accuracy thresholds derived from frozen capture outputs."""

from collections import defaultdict
import math

import numpy as np


CAPTURE = 'capture_vs_fp64'
CANDIDATE = 'candidate_vs_fp64'
KEY_COLUMNS = ('input', 'direction', 'source_event', 'i', 'head')
METRICS = ('relative_l2', 'max_abs')
BASELINE_VERSION = 1


def row_key(row):
  return tuple(row[field] for field in KEY_COLUMNS)


def integrity_errors(rows, expected, *, failures, inputs_unchanged):
  observed = {(row_key(row), row['comparison']) for row in rows}
  reasons = []
  if observed != expected or len(observed) != len(rows):
    reasons.append('missing, duplicate, or unexpected metric rows')
  if failures:
    reasons.append('replay or source verification failed')
  if not inputs_unchanged:
    reasons.append('source hashes changed')
  if any(row['nonfinite_count'] for row in rows):
    reasons.append('nonfinite output')
  return reasons


def group_rows(rows, comparison):
  groups = defaultdict(list)
  for row in rows:
    if row['comparison'] == comparison:
      groups[(row['input'], row['direction'], row['mask_group'])].append(row)
  return groups


def prepare(rows, source_hashes):
  groups = []
  for (name, direction, mask_group), items in sorted(group_rows(rows, CAPTURE).items()):
    worst = {
        metric: max((row[metric] for row in items if row[metric] is not None),
                    default=0.0)
        for metric in METRICS
    }
    groups.append(dict(
        input=name, direction=direction, mask_group=mask_group,
        capture_worst=worst, head_rows=len(items),
    ))
  return dict(version=BASELINE_VERSION, source_hashes=source_hashes, groups=groups)


def statistics(rows, metric):
  defined = [row for row in rows if row[metric] is not None]
  if not defined:
    return dict(worst=None, median=None, p95=None)
  values = [row[metric] for row in defined]
  return dict(
      worst=max(defined, key=lambda row: row[metric]),
      median=float(np.median(values)), p95=float(np.percentile(values, 95)),
  )


def evaluate(rows, source_hashes, baseline):
  if baseline['version'] != BASELINE_VERSION:
    raise ValueError('unsupported accuracy baseline version')
  if any(baseline['source_hashes'].get(path) != digest
         for path, digest in source_hashes.items()):
    raise ValueError('accuracy baseline source hashes do not match replay data')

  limits_by_group = {}
  for group in baseline['groups']:
    key = (group['input'], group['direction'], group['mask_group'])
    worst = group['capture_worst']
    if key in limits_by_group or any(
        not math.isfinite(worst[metric]) or worst[metric] < 0 for metric in METRICS
    ):
      raise ValueError(f'invalid accuracy baseline group {key}')
    limits_by_group[key] = dict(
        relative_l2=min(0.01, 1.2 * worst['relative_l2']),
        max_abs=1.2 * worst['max_abs'],
    )

  results = []
  for key, items in sorted(group_rows(rows, CANDIDATE).items()):
    if key not in limits_by_group:
      raise ValueError(f'accuracy baseline is missing group {key}')
    limits = limits_by_group[key]
    violations = [
        dict(metric=metric, row=row)
        for row in items for metric, limit in limits.items()
        if row[metric] is not None and row[metric] > limit
    ]
    results.append(dict(
        input=key[0], direction=key[1], mask_group=key[2], thresholds=limits,
        metrics={metric: statistics(items, metric) for metric in METRICS},
        passed=not violations, violations=violations,
    ))
  violation_count = sum(len(group['violations']) for group in results)
  passed = bool(results) and violation_count == 0
  return dict(
      status='passed' if passed else 'failed', passed=passed,
      violation_count=violation_count, groups=results,
  )
