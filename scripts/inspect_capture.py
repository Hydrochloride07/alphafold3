"""Validate a captured run or inspect one event without allowing pickle."""

import argparse
import json
from pathlib import Path

from alphafold3.model.capture import load_event

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('path', type=Path, help='Run directory or event .json/.npz')
args = parser.parse_args()
if args.path.is_dir():
  manifest = json.loads((args.path / 'manifest.json').read_text())
  seen = set()
  for line in (args.path / manifest['event_index']).read_text().splitlines():
    event = json.loads(line)
    if event['event_id'] in seen:
      raise ValueError('Duplicate event ID: ' + event['event_id'])
    seen.add(event['event_id'])
    record, tensors = load_event(args.path / event['json'])
    if record['event_id'] != event['event_id']:
      raise ValueError('Index/event ID mismatch')
  if manifest['status'] == 'complete' and len(seen) != manifest['event_count']:
    raise ValueError('Manifest event count does not match index')
  print(json.dumps(manifest, indent=2))
  print(f'Validated {len(seen)} events; status={manifest["status"]}')
else:
  record, tensors = load_event(args.path)
  print(json.dumps(record, indent=2))
