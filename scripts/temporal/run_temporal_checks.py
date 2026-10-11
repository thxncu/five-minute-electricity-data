"""Reproduce exact arithmetic, temporal reference models, and change screening."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, default=here.parents[1])
parser.add_argument('--output-dir', type=Path, default=None)
args = parser.parse_args()
os.environ['KPX_RAW_DIR'] = str(args.package_root / 'data_raw')
os.environ['KPX_TEMPORAL_OUTPUT'] = str(args.output_dir or args.package_root / 'reproduced_results' / 'temporal')
for script in ['rounding_analysis.py', 'markov_analysis.py', 'transition_analysis.py', 'figures.py']:
    subprocess.run([sys.executable, str(here / script)], check=True)
