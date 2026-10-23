"""Check the editable installation, compiled extension and chemical data."""
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
os.environ['JAX_PLATFORMS'] = 'cuda' if args.gpu else 'cpu'

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
import alphafold3.cpp
import jax

root = Path(__file__).resolve().parents[1]
source = Path(folding_input.__file__).resolve()
assert source.is_relative_to(root / 'src'), source
ccd = chemical_components.Ccd()
assert 'ALA' in ccd
print('Python:', sys.executable)
print('Source:', source)
print('C++ extension:', alphafold3.cpp.__file__)
print('CCD entries:', len(ccd))
devices = jax.devices('gpu' if args.gpu else 'cpu')
print('JAX:', jax.__version__, 'Devices:', devices)
if args.gpu:
    for device in devices:
        print('GPU:', device.device_kind, 'CC:', device.compute_capability)
