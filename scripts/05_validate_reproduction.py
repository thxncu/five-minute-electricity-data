from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduced_results"
EXPECTED = ROOT / "expected_results"
VALID = ROOT / "validation"
DERIVED = ROOT / "derived_data"

rows = []

def add(item, status, detail):
    rows.append({"item":item,"status":status,"detail":detail})

# Key claims
expected_key = json.loads((VALID / "expected_key_results.json").read_text(encoding="utf-8"))
actual_key = json.loads((REPRO / "key_results.json").read_text(encoding="utf-8"))
for k, expected in expected_key.items():
    actual = actual_key.get(k)
    ok = np.isclose(float(actual), float(expected), atol=1e-9, rtol=0)
    add(f"key:{k}", "PASS" if ok else "FAIL", f"expected={expected}; actual={actual}")


# Mission-specific claims
mission_expected_path = VALID / "mission_expected_key_results.json"
mission_actual_path = REPRO / "mission_key_results.json"
if not mission_expected_path.exists() or not mission_actual_path.exists():
    add("mission:key_results", "FAIL", "Missing mission-specific expected or reproduced key-results JSON.")
else:
    mission_expected = json.loads(mission_expected_path.read_text(encoding="utf-8"))
    mission_actual = json.loads(mission_actual_path.read_text(encoding="utf-8"))
    for k, expected in mission_expected.items():
        actual = mission_actual.get(k)
        if actual is None:
            add(f"mission_key:{k}", "FAIL", "Key missing from reproduced mission-specific results.")
            continue
        ok = np.isclose(float(actual), float(expected), atol=1e-9, rtol=0)
        add(f"mission_key:{k}", "PASS" if ok else "FAIL", f"expected={expected}; actual={actual}")

# Compare every expected CSV; missing output is a failure.
for expected_path in sorted((EXPECTED / "tables").glob("*.csv")):
    actual_path = REPRO / "tables" / expected_path.name
    if not actual_path.exists():
        add(f"table:{expected_path.name}", "FAIL", "Expected table was not generated.")
        continue
    try:
        e = pd.read_csv(expected_path)
        a = pd.read_csv(actual_path)
        if list(e.columns) != list(a.columns) or len(e) != len(a):
            add(f"table:{expected_path.name}", "FAIL", f"shape/columns differ: expected {e.shape}, actual {a.shape}")
            continue
        okay = True
        max_diff = 0.0
        for col in e.columns:
            if pd.api.types.is_numeric_dtype(e[col]) and pd.api.types.is_numeric_dtype(a[col]):
                ev = pd.to_numeric(e[col], errors="coerce").to_numpy(float)
                av = pd.to_numeric(a[col], errors="coerce").to_numpy(float)
                diff = np.abs(ev-av)
                if np.isfinite(diff).any(): max_diff=max(max_diff,float(np.nanmax(diff)))
                if not np.all(np.isclose(ev,av,atol=1e-8,rtol=1e-10,equal_nan=True)): okay=False
            else:
                es=e[col].fillna("").astype(str).str.strip()
                ass=a[col].fillna("").astype(str).str.strip()
                if not es.equals(ass): okay=False
        add(f"table:{expected_path.name}", "PASS" if okay else "FAIL", f"rows={len(e)}; max_numeric_diff={max_diff}")
    except Exception as exc:
        add(f"table:{expected_path.name}", "FAIL", repr(exc))

# Check that each expected figure was generated.
for fig in sorted(p for p in (EXPECTED / "figures").iterdir() if p.suffix in {".png", ".pdf", ".eps"}):
    actual = REPRO / "figures" / fig.name
    add(f"figure:{fig.name}", "PASS" if actual.exists() and actual.stat().st_size > 1000 else "FAIL", f"generated={actual.exists()}")

report = pd.DataFrame(rows)
(REPRO / "validation").mkdir(parents=True, exist_ok=True)
report.to_csv(REPRO / "validation" / "baseline_reference_comparison.csv", index=False)
failures = report[~report.status.eq("PASS")]
print(report.status.value_counts().to_string())
if len(failures):
    print("\nFailures:\n", failures.to_string(index=False))
    sys.exit(1)
print("\nReference-output checks passed.")
