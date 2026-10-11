"""Reproduce all analyses from local source files and compare fixed references."""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Use Python 3.12 with the supplied pinned requirements.")

# The derived directories contain generated results only. A clean run prevents
# an output from an earlier execution from masking a missing current result.
for name in ["derived_data", "reproduced_results"]:
    path = ROOT / name
    if path.is_symlink():
        raise SystemExit(f"Generated output directory must not be a symlink: {path}")
    if path.exists():
        shutil.rmtree(path)

environment = os.environ.copy()
environment.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg",
                    "PYTHONTZPATH": "", "PYTHONDONTWRITEBYTECODE": "1"})
steps = [
    ROOT / "scripts" / "01_build_audited_panel.py",
    ROOT / "scripts" / "02_reproduce_core_results.py",
    ROOT / "scripts" / "03_generate_additional_results_and_figures.py",
    ROOT / "scripts" / "04_evaluate_decision_usability.py",
    ROOT / "scripts" / "06_downstream_forecast_benchmark.py",
    ROOT / "scripts" / "temporal" / "run_temporal_checks.py",
    ROOT / "scripts" / "external_schema_check.py",
    ROOT / "scripts" / "05_validate_reproduction.py",
    ROOT / "scripts" / "07_validate_extended_reproduction.py",
]
for step in steps:
    print(f"Running {step.relative_to(ROOT)}", flush=True)
    subprocess.run([sys.executable, str(step)], cwd=ROOT, env=environment, check=True)
    print(f"Completed {step.relative_to(ROOT)}", flush=True)
print("All analyses and scientific reference comparisons completed successfully.", flush=True)
