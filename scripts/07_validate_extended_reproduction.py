"""Compare downstream, temporal and external-schema scientific outputs.

Missing expected outputs fail. Numerical tables, including every saved test
prediction, are compared with fixed references. Runtime and interpreter patch
metadata are excluded from the recursive scientific-JSON comparison.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "expected_results"
ACTUAL = ROOT / "reproduced_results"
REPORT = ACTUAL / "validation"
REPORT.mkdir(parents=True, exist_ok=True)

ATOL = 1e-7
RTOL = 1e-11
IGNORED_JSON_KEYS = {"elapsed_seconds", "runtime_seconds", "software", "output_dir"}
rows = []
numeric_cells = 0
scientific_json_leaves = 0


def add(item, passed, detail):
    rows.append({"item": item, "status": "PASS" if passed else "FAIL", "detail": detail})


def compare_table(reference, generated):
    global numeric_cells
    if not generated.exists():
        return False, "Expected table is missing."
    e = pd.read_csv(reference)
    a = pd.read_csv(generated)
    if list(e.columns) != list(a.columns) or len(e) != len(a):
        return False, f"Table shape or columns differ: reference={e.shape}, generated={a.shape}."
    maximum = 0.0
    different = []
    for column in e.columns:
        if pd.api.types.is_numeric_dtype(e[column]) and pd.api.types.is_numeric_dtype(a[column]):
            ev = e[column].to_numpy(dtype=float)
            av = a[column].to_numpy(dtype=float)
            numeric_cells += len(ev)
            if pd.api.types.is_integer_dtype(e[column]) or pd.api.types.is_bool_dtype(e[column]):
                good = np.array_equal(ev, av, equal_nan=True)
            else:
                good = bool(np.all(np.isclose(ev, av, atol=ATOL, rtol=RTOL, equal_nan=True)))
            finite = np.isfinite(ev) & np.isfinite(av)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(ev[finite] - av[finite]))))
            if not good:
                different.append(column)
        elif not e[column].fillna("").astype(str).equals(a[column].fillna("").astype(str)):
            different.append(column)
    detail = f"rows={len(e)}; columns={len(e.columns)}; max_numeric_difference={maximum:.12g}"
    if different:
        detail += "; differing_columns=" + ", ".join(different)
    return not different, detail


def compare_json_value(reference, generated, path="root"):
    global scientific_json_leaves
    if isinstance(reference, dict):
        if not isinstance(generated, dict):
            return [f"{path}: expected an object"]
        ek = set(reference) - IGNORED_JSON_KEYS
        ak = set(generated) - IGNORED_JSON_KEYS
        problems = []
        if ek != ak:
            problems.append(f"{path}: scientific keys differ; missing={sorted(ek-ak)}; extra={sorted(ak-ek)}")
        for key in sorted(ek & ak):
            problems.extend(compare_json_value(reference[key], generated[key], f"{path}.{key}"))
        return problems
    if isinstance(reference, list):
        if not isinstance(generated, list) or len(reference) != len(generated):
            return [f"{path}: list lengths or types differ"]
        problems = []
        for i, (e, a) in enumerate(zip(reference, generated)):
            problems.extend(compare_json_value(e, a, f"{path}[{i}]"))
        return problems
    scientific_json_leaves += 1
    if isinstance(reference, bool) or reference is None or isinstance(reference, str):
        return [] if generated == reference and type(generated) is type(reference) else [f"{path}: {reference!r} != {generated!r}"]
    if isinstance(reference, int):
        return [] if generated == reference else [f"{path}: {reference!r} != {generated!r}"]
    if isinstance(reference, float):
        try:
            good = (math.isnan(reference) and math.isnan(float(generated))) or math.isclose(reference, float(generated), abs_tol=ATOL, rel_tol=RTOL)
        except (TypeError, ValueError):
            good = False
        return [] if good else [f"{path}: {reference!r} != {generated!r}"]
    return [] if generated == reference else [f"{path}: values differ"]


def compare_json(reference, generated):
    if not generated.exists():
        return False, "Expected scientific JSON is missing."
    e = json.loads(reference.read_text(encoding="utf-8"))
    a = json.loads(generated.read_text(encoding="utf-8"))
    problems = compare_json_value(e, a)
    return not problems, "; ".join(problems[:8]) if problems else "All scientific keys and values match."


def verify_sources():
    manifest = pd.read_csv(EXPECTED / "tables" / "source_file_manifest_sha256.csv")
    for row in manifest.itertuples():
        source = ROOT / "data_raw" / row.file
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else "missing"
        add(f"source:KPX:{row.file}", digest == row.sha256, f"sha256={digest}")
    ext = ROOT / "data_raw_external" / "caiso"
    manifest = json.loads((ext / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        source = ext / row["filename"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else "missing"
        add(f"source:CAISO:{row['filename']}", digest == row["sha256"], f"sha256={digest}")


verify_sources()
for module in ["downstream", "temporal", "external_schema"]:
    expected_directory = EXPECTED / module
    for reference in sorted(expected_directory.rglob("*")):
        if not reference.is_file():
            continue
        relative = reference.relative_to(EXPECTED)
        generated = ACTUAL / relative
        try:
            if reference.name.endswith((".csv", ".csv.gz")):
                passed, detail = compare_table(reference, generated)
            elif reference.suffix == ".json":
                passed, detail = compare_json(reference, generated)
            elif reference.suffix in {".png", ".pdf", ".eps"}:
                passed = generated.exists() and generated.stat().st_size > 1000
                detail = "Scientific figure regenerated." if passed else "Expected scientific figure is missing or empty."
            else:
                passed, detail = False, "Unexpected file type in scientific reference directory."
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        add(f"output:{relative.as_posix()}", passed, detail)

# Cross-output checks ensure that the forecast denominators and the source
# flags remain consistent across independently implemented analyses.
try:
    prediction = pd.read_csv(ACTUAL / "downstream" / "test_predictions.csv.gz", parse_dates=["origin", "target_time"])
    common = prediction["common_evaluation_origin"]
    all_issued = prediction[["issued_raw", "issued_strict", "issued_permissive", "issued_causal_imputation"]].all(axis=1)
    reference_usable = prediction["target_reference_usable"]
    add("cross:common_prediction_cohort", (common == (all_issued & reference_usable)).all(),
        f"common_n={int(common.sum())}; evaluable_target_n={int(reference_usable.sum())}")
    add("cross:forecast_horizon", ((prediction.target_time - prediction.origin).dt.total_seconds() == 3600).all(),
        "Every saved target follows its origin by exactly one hour.")
    add("cross:policy_inclusion", ((~prediction.issued_strict | prediction.issued_permissive) &
                                  (~prediction.issued_permissive | prediction.issued_raw)).all(),
        "Strict issued origins are a subset of permissive origins, which are a subset of raw origins.")
    holdout = pd.read_csv(ACTUAL / "temporal" / "markov_chronological_holdout.csv")
    cadence = holdout[(holdout.mission_code == "cadence_delivery") & (holdout.window_minutes == 60)].iloc[0]
    add("cross:temporal_and_forecast_endings", int(cadence.test_eligible_endings_with_prehistory) - len(prediction) == 12,
        "The forecast horizon removes the final twelve five-minute origins; temporal history endings retain them.")
    panel = pd.read_csv(ACTUAL / "tables" / "quality_status_panel.csv", usecols=["flag_supply_identity"])
    forecast = json.loads((ACTUAL / "downstream" / "run_summary.json").read_text(encoding="utf-8"))
    exact = pd.read_csv(ACTUAL / "temporal" / "supply_identity_tolerance_exact_decimal.csv")
    count_exact = int(exact.loc[pd.to_numeric(exact.tolerance_mw).eq(1.0), "records_exceeding_tolerance"].iloc[0])
    count_baseline = int(panel.flag_supply_identity.sum())
    count_forecast = int(forecast["quality_flag_counts"]["supply_identity"])
    add("cross:one_mw_identity_rule", count_exact == count_baseline == count_forecast == 6,
        f"exact={count_exact}; baseline={count_baseline}; forecasting={count_forecast}")
except Exception as exc:
    add("cross:scientific_consistency", False, f"{type(exc).__name__}: {exc}")

report = pd.DataFrame(rows)
report.to_csv(REPORT / "extended_reference_comparison.csv", index=False)
failures = report[report.status != "PASS"]
summary = {"comparisons": len(report), "passed": int((report.status == "PASS").sum()),
           "failed": int(len(failures)), "numeric_table_cells_compared": int(numeric_cells),
           "scientific_json_leaves_compared": int(scientific_json_leaves),
           "absolute_numeric_tolerance": ATOL, "relative_numeric_tolerance": RTOL,
           "ignored_json_keys": sorted(IGNORED_JSON_KEYS)}
(REPORT / "extended_reference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
if len(failures):
    print(failures.to_string(index=False), flush=True)
    sys.exit(1)
print("Extended scientific reference comparisons passed.", flush=True)
