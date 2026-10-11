"""Reproduce the retrospective one-hour demand forecasting benchmark.

All inputs are local quarterly KPX CSV files. No network access is performed.
The script can be copied to Reproducibility_Package/scripts and run there, or
run with --package-root and --output-dir. Dates are Korean Standard Time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

# Bound thread use and make it explicit in the reproducible configuration.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

SEED = 20261011
FIELDS = ["supply_capacity_mw", "current_demand_mw", "maximum_forecast_demand_mw",
          "supply_reserve_mw", "supply_reserve_rate_pct", "operating_reserve_mw",
          "operating_reserve_rate_pct"]
COLMAP = dict(zip(["공급능력(MW)", "현재수요(MW)", "최대예측수요(MW)",
                   "공급예비력(MW)", "공급예비율(%)", "운영예비력(MW)",
                   "운영예비율(%)"], FIELDS))
POLICIES = ["raw", "strict", "permissive", "causal_imputation"]
WINDOW = 12
HORIZON = 12
TRAIN_END = pd.Timestamp("2025-07-01")
TEST_START = pd.Timestamp("2026-01-01")
PARAMS = dict(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31,
              min_samples_leaf=40, l2_regularization=1.0, max_bins=255,
              early_stopping=False, random_state=SEED)
CANDIDATES = [100, 200, 300]


def metrics(y, p):
    if len(y) == 0:
        return dict(n_scored=0, mae_mw=None, rmse_mw=None, mape_pct=None)
    e = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return dict(n_scored=int(len(e)), mae_mw=float(np.mean(np.abs(e))),
                rmse_mw=float(np.sqrt(np.mean(e * e))),
                mape_pct=float(100.0 * np.mean(np.abs(e / y))))


def build_panel(root):
    manifest = pd.read_csv(root / "expected_results/tables/source_file_manifest_sha256.csv")
    checks = []
    parts = []
    for _, row in manifest.iterrows():
        p = root / "data_raw" / row["file"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != row["sha256"]:
            raise ValueError(f"Source hash mismatch: {p.name}")
        d = pd.read_csv(p, encoding="cp949").rename(columns=COLMAP)
        d["timestamp"] = pd.to_datetime(d["기준일시"].astype(str), format="%Y%m%d%H%M%S")
        parts.append(d[["timestamp", *FIELDS]])
        checks.append(dict(file=p.name, sha256=h, rows=int(len(d))))
    obs = pd.concat(parts).sort_values("timestamp").set_index("timestamp")
    assert len(obs) == 209323 and not obs.index.duplicated().any()
    ix = pd.date_range("2024-07-01", "2026-06-30 23:55", freq="5min")
    grid = obs.reindex(ix)
    grid.index.name = "timestamp"
    observed = grid[FIELDS].notna().all(axis=1)
    residual = grid.supply_reserve_mw - (grid.supply_capacity_mw - grid.current_demand_mw)
    flags = pd.DataFrame(index=ix)
    flags["forecast_zero"] = observed & grid.maximum_forecast_demand_mw.eq(0)
    flags["supply_identity"] = observed & residual.abs().gt(1.0)
    flags["operating_zero"] = observed & grid.operating_reserve_mw.eq(0)
    flags["low_operating_high_supply"] = observed & grid.operating_reserve_mw.lt(1500) & grid.supply_reserve_mw.ge(5500)
    flags["nesting_ambiguity"] = observed & (grid.operating_reserve_mw - grid.supply_reserve_mw).gt(1e-9)
    flags["unresolved_transition"] = observed & ix.to_series(index=ix).between("2026-06-30 16:20", "2026-06-30 17:10")
    block_cols = ["forecast_zero", "supply_identity", "operating_zero", "low_operating_high_supply", "unresolved_transition"]
    permissive = observed & ~flags[block_cols].any(axis=1)
    strict = permissive & ~flags.nesting_ambiguity
    # Imputation reconstructs only the fields implicated by the declared rule.
    # All operands of a failed identity are masked because the faulty operand
    # cannot be identified from that identity alone. Ambiguity stays flagged.
    mask = grid[FIELDS].isna()
    mask.loc[flags.forecast_zero, "maximum_forecast_demand_mw"] = True
    mask.loc[flags.supply_identity, FIELDS[:2] + FIELDS[3:5]] = True
    op_mask = flags[["operating_zero", "low_operating_high_supply", "unresolved_transition"]].any(axis=1)
    mask.loc[op_mask, FIELDS[5:]] = True
    cleaned = grid[FIELDS].mask(mask)
    filled = cleaned.ffill()
    positions = pd.Series(np.arange(len(ix)), index=ix)
    ages = pd.DataFrame(index=ix)
    for f in FIELDS:
        source = positions.where(~mask[f]).ffill()
        assert np.all(source.dropna().to_numpy() <= positions[source.notna()].to_numpy())
        ages[f] = 5 * (positions - source)
    point = dict(raw=observed, strict=strict, permissive=permissive,
                 causal_imputation=filled.notna().all(axis=1))
    windows = {p: a.astype(int).rolling(WINDOW, min_periods=WINDOW).sum().eq(WINDOW).to_numpy()
               for p, a in point.items()}
    # Reference demand is never imputed. Supply-identity failures are excluded
    # equally from every target set, without choosing a faulty identity operand.
    reference_ok = observed & grid.current_demand_mw.gt(0) & ~flags.supply_identity
    return grid, filled, flags, mask, ages, windows, reference_ok, checks


def feature_matrix(values, index):
    # The first block is time t, and the last is t-55 minutes.
    arr = values.to_numpy(dtype=np.float32)
    blocks = []
    for lag in range(WINDOW):
        a = np.full_like(arr, np.nan)
        if lag:
            a[lag:] = arr[:-lag]
        else:
            a[:] = arr
        blocks.append(a)
    # The forecast target's calendar is known at the forecast origin.
    target_ix = index + pd.Timedelta(hours=1)
    hour = target_ix.hour + target_ix.minute / 60.0
    week = target_ix.dayofweek.to_numpy()
    year = target_ix.dayofyear.to_numpy() - 1 + hour / 24.0
    cal = np.column_stack([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                           np.sin(2*np.pi*week/7), np.cos(2*np.pi*week/7),
                           np.sin(2*np.pi*year/365.25), np.cos(2*np.pi*year/365.25)]).astype(np.float32)
    return np.column_stack([*blocks, cal])


def aggregate_bootstrap(frame, common, n_boot=2000):
    # Moving blocks of seven consecutive calendar days preserve within-day
    # overlap and short-range dependence. Models are not refitted per draw.
    df = frame.loc[common].copy()
    day_index = pd.date_range(frame.origin.min().normalize(), frame.origin.max().normalize(), freq="D")
    y = df.target_demand_mw.to_numpy()
    perday = {}
    for p in POLICIES + ["persistence"]:
        e = df[f"prediction_{p}_mw"].to_numpy() - y
        tmp = pd.DataFrame(dict(day=df.origin.dt.normalize().to_numpy(), count=1,
                                abs_error=np.abs(e), sq_error=e*e, ape=100*np.abs(e/y)))
        perday[p] = tmp.groupby("day").sum().reindex(day_index, fill_value=0).to_numpy()
    rng = np.random.default_rng(SEED)
    n_days = len(day_index)
    block_length = 7
    n_blocks = int(np.ceil(n_days / block_length))
    starts = rng.integers(0, n_days - block_length + 1, size=(n_boot, n_blocks))
    sample = (starts[..., None] + np.arange(block_length)).reshape(n_boot, -1)[:, :n_days]
    boot_metrics = {}
    for p, a in perday.items():
        totals = a[sample].sum(axis=1)
        boot_metrics[p] = dict(mae_mw=totals[:,1]/totals[:,0],
                               rmse_mw=np.sqrt(totals[:,2]/totals[:,0]),
                               mape_pct=totals[:,3]/totals[:,0])
    pairs = [("strict", "raw"), ("permissive", "strict"),
             ("causal_imputation", "permissive"), ("raw", "persistence")]
    rows = []
    for a,b in pairs:
        ma = metrics(y, df[f"prediction_{a}_mw"])
        mb = metrics(y, df[f"prediction_{b}_mw"])
        for metric in ["mae_mw", "rmse_mw", "mape_pct"]:
            v = boot_metrics[a][metric] - boot_metrics[b][metric]
            rows.append(dict(comparison=f"{a} minus {b}", metric=metric,
                             observed_difference=ma[metric]-mb[metric],
                             lower_95=float(np.quantile(v,.025)), upper_95=float(np.quantile(v,.975)),
                             n_common=int(common.sum()), blocks_days=block_length, bootstrap_draws=n_boot,
                             interpretation="Conditional paired error difference; trained models fixed"))
    return pd.DataFrame(rows)


def summarize_flagged_episodes(prediction_path, out):
    """Describe rejected complete histories and their contiguous origin runs."""
    frame = pd.read_csv(prediction_path, parse_dates=["origin", "target_time"])
    selected = frame.issued_raw & ~frame.issued_permissive & frame.target_reference_usable
    cohort = frame.loc[selected].sort_values("origin", kind="mergesort").copy()
    if cohort.empty or cohort.origin.duplicated().any():
        raise ValueError("The flagged-history cohort must contain unique forecast origins.")
    predicted = ["prediction_raw_mw", "prediction_causal_imputation_mw"]
    if not np.isfinite(cohort[["target_demand_mw", *predicted]].to_numpy()).all():
        raise ValueError("Both predictors and the reference target must be finite on the cohort.")
    if not cohort.target_demand_mw.gt(0).all():
        raise ValueError("Percentage errors require positive reference demand.")
    cohort["run_id"] = cohort.origin.diff().ne(pd.Timedelta(minutes=5)).cumsum()
    blocking_reasons = ["forecast_zero", "supply_identity", "operating_zero",
                        "low_operating_high_supply", "unresolved_transition"]

    def describe(group):
        result = dict(n_origins=int(len(group)),
                      n_contiguous_runs=int(group.run_id.nunique()),
                      n_calendar_days=int(group.origin.dt.normalize().nunique()),
                      origin_start_kst=str(group.origin.min()), origin_end_kst=str(group.origin.max()),
                      target_start_kst=str(group.target_time.min()), target_end_kst=str(group.target_time.max()),
                      blocking_reasons=";".join(reason for reason in blocking_reasons
                                               if group[f"history_{reason}_count"].gt(0).any()))
        y = group.target_demand_mw.to_numpy()
        for policy in ["raw", "causal_imputation"]:
            pred = group[f"prediction_{policy}_mw"].to_numpy()
            score = metrics(y, pred)
            for name in ["rmse_mw", "mae_mw", "mape_pct"]:
                result[f"{policy}_{name}"] = score[name]
            result[f"{policy}_sse_mw2"] = float(np.sum((pred - y) ** 2))
        for name in ["rmse_mw", "mae_mw", "mape_pct", "sse_mw2"]:
            result[f"raw_minus_imputed_{name}"] = result[f"raw_{name}"] - result[f"causal_imputation_{name}"]
        result["rmse_reduction_pct"] = 100 * result["raw_minus_imputed_rmse_mw"] / result["raw_rmse_mw"]
        return result

    total = describe(cohort)

    def shares(row):
        row["raw_sse_share_pct"] = 100 * row["raw_sse_mw2"] / total["raw_sse_mw2"]
        row["imputed_sse_share_pct"] = 100 * row["causal_imputation_sse_mw2"] / total["causal_imputation_sse_mw2"]
        change = total["raw_minus_imputed_sse_mw2"]
        row["sse_reduction_share_pct"] = 100 * row["raw_minus_imputed_sse_mw2"] / change if change else None
        return row

    runs = pd.DataFrame([shares(dict(run_id=int(run), **describe(group)))
                         for run, group in cohort.groupby("run_id", sort=True)])
    # Selection uses observed test errors and is descriptive, not prospective.
    largest_run = int(runs.loc[runs.raw_sse_mw2.idxmax(), "run_id"])
    runs["posthoc_largest_raw_sse_run"] = runs.run_id.eq(largest_run)
    runs.to_csv(out / "flagged_history_runs.csv", index=False)
    partitions = [("all_rejected_complete_histories", cohort),
                  ("posthoc_largest_raw_sse_run", cohort[cohort.run_id.eq(largest_run)]),
                  ("remaining_origins", cohort[cohort.run_id.ne(largest_run)])]
    concentration = pd.DataFrame([shares(dict(cohort=name, **describe(group))) for name, group in partitions])
    concentration.to_csv(out / "flagged_history_concentration.csv", index=False)
    reason_rows = []
    for reason in blocking_reasons:
        group = cohort[cohort[f"history_{reason}_count"].gt(0)]
        if len(group):
            reason_rows.append(shares(dict(blocking_reason=reason, **describe(group))))
    pd.DataFrame(reason_rows).to_csv(out / "flagged_history_reason_groups.csv", index=False)
    multiple = cohort[[f"history_{reason}_count" for reason in blocking_reasons]].gt(0).sum(axis=1).gt(1)
    definition = dict(
        input_table=prediction_path.name,
        cohort_definition="Evaluable target origins with complete raw histories that fail permissive acceptance",
        timezone="Korea Standard Time (UTC+09:00)",
        run_definition="Maximal consecutive five-minute origins within the selected cohort; a gap starts a new run",
        observation_unit="Forecast origin with an overlapping twelve-point input history; a run is not an independently verified physical event",
        largest_run_selection="Largest total raw squared error among the observed test-cohort runs, selected after observing errors; chronological order resolves ties",
        n_origins=int(len(cohort)), n_runs=int(len(runs)),
        n_calendar_days=int(cohort.origin.dt.normalize().nunique()),
        largest_raw_sse_run_id=largest_run,
        origins_with_multiple_blocking_reasons=int(multiple.sum()),
        squared_error_unit="MW squared",
        sse_reduction_share_definition="Run raw-minus-imputed squared error divided by the corresponding whole-cohort difference; shares can be negative or exceed 100 percent",
        limitations=["Descriptive subgroup decomposition without a subgroup significance test",
                     "The largest-error run is selected from the test errors and does not define a prospective event detector",
                     "Policy-specific training samples and overlapping histories prevent a causal interpretation",
                     "Published reference demand is screened but is not independent physical ground truth",
                     "Pooled differences do not establish uniform benefits across flagged events"],
        tables=["flagged_history_runs.csv", "flagged_history_concentration.csv", "flagged_history_reason_groups.csv"])
    (out / "flagged_history_episode_definition.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    return concentration


def main():
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--postprocess-only", action="store_true",
                    help="Describe flagged-history episodes from saved predictions without fitting models.")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Input table for --postprocess-only; defaults to test_predictions.csv.gz in the output directory.")
    args = ap.parse_args()
    out = args.output_dir or args.package_root / "reproduced_results/downstream"
    out.mkdir(parents=True, exist_ok=True)
    if args.predictions is not None and not args.postprocess_only:
        ap.error("--predictions requires --postprocess-only")
    if args.postprocess_only:
        result = summarize_flagged_episodes(args.predictions or out / "test_predictions.csv.gz", out)
        print(result[["cohort", "n_origins", "raw_rmse_mw", "causal_imputation_rmse_mw"]].to_string(index=False), flush=True)
        return
    grid, filled, flags, imputed, ages, windows, ref_ok, hashes = build_panel(args.package_root)
    ix = grid.index
    target_times = ix + pd.Timedelta(hours=1)
    y = grid.current_demand_mw.shift(-HORIZON).to_numpy()
    target_ok = ref_ok.shift(-HORIZON, fill_value=False).to_numpy()
    train = (ix >= ix[WINDOW-1]) & (target_times < TRAIN_END) & target_ok
    validation = (ix >= TRAIN_END) & (target_times < TEST_START) & target_ok
    refit = (ix >= ix[WINDOW-1]) & (target_times < TEST_START) & target_ok
    scheduled_test = (ix >= TEST_START) & (target_times <= ix[-1])
    eval_test = scheduled_test & target_ok
    assert not np.any(train & validation) and not np.any(refit & scheduled_test)
    assert target_times[train].max() < ix[validation].min()
    assert target_times[refit].max() < ix[scheduled_test].min()
    x_raw = feature_matrix(grid[FIELDS], ix)
    x_filled = feature_matrix(filled, ix)
    all_common = np.logical_and.reduce([windows[p] for p in POLICIES])
    common_val = validation & all_common
    common_test = eval_test & all_common
    selection_rows = []
    ncounts = []
    # One common iteration choice is selected by mean validation RMSE across
    # all four policies on exactly the same validation origins.
    for policy in POLICIES:
        X = x_filled if policy == "causal_imputation" else x_raw
        use = train & windows[policy]
        base = X[:,1].astype(float)
        assert np.isfinite(X[use]).all()
        model = HistGradientBoostingRegressor(max_iter=max(CANDIDATES), **PARAMS)
        print(f"Validation fit: {policy}; n={use.sum()}", flush=True)
        with threadpool_limits(limits=2):
            model.fit(X[use], y[use]-base[use])
            for stage, residual_pred in enumerate(model.staged_predict(X[common_val]), start=1):
                if stage in CANDIDATES:
                    row = dict(policy=policy, max_iter=stage, **metrics(y[common_val], residual_pred+base[common_val]))
                    selection_rows.append(row)
        ncounts.append(dict(policy=policy, train_n=int(use.sum()), validation_n=int((validation & windows[policy]).sum()),
                            common_validation_n=int(common_val.sum()), refit_n=int((refit & windows[policy]).sum())))
    select = pd.DataFrame(selection_rows)
    select.to_csv(out / "validation_model_selection.csv", index=False)
    choice = select.groupby("max_iter").rmse_mw.mean().sort_values().index[0]
    choice = int(choice)
    print(f"Selected common max_iter={choice}", flush=True)
    test_idx = np.flatnonzero(scheduled_test)
    frame = pd.DataFrame(dict(origin=ix[test_idx], target_time=target_times[test_idx],
                              target_demand_mw=y[test_idx], target_reference_usable=target_ok[test_idx],
                              common_evaluation_origin=common_test[test_idx]))
    frame["prediction_persistence_mw"] = x_raw[test_idx,1]
    score_rows = []
    month_rows = []
    for policy in POLICIES:
        X = x_filled if policy == "causal_imputation" else x_raw
        use = refit & windows[policy]
        issue = scheduled_test & windows[policy]
        base = X[:,1].astype(float)
        model = HistGradientBoostingRegressor(max_iter=choice, **PARAMS)
        print(f"Final fit: {policy}; n={use.sum()}; issued={issue.sum()}", flush=True)
        predictions = np.full(len(ix), np.nan)
        with threadpool_limits(limits=2):
            model.fit(X[use], y[use]-base[use])
            predictions[issue] = model.predict(X[issue]) + base[issue]
        frame[f"issued_{policy}"] = windows[policy][test_idx]
        frame[f"prediction_{policy}_mw"] = predictions[test_idx]
        for pool, m in [("policy_issued", eval_test & windows[policy]), ("common_intersection", common_test)]:
            row = dict(policy=policy, model="histogram_gradient_boosting", evaluation_pool=pool,
                       scheduled_origins=int(scheduled_test.sum()), issued_origins=int(issue.sum()),
                       prediction_coverage_pct=100.0*float(issue.sum()/scheduled_test.sum()),
                       available_target_origins=int(eval_test.sum()),
                       scorable_prediction_coverage_pct=100.0*float((eval_test & windows[policy]).sum()/eval_test.sum()),
                       **metrics(y[m], predictions[m]))
            score_rows.append(row)
            baseline_pred = base[m] if policy == "causal_imputation" else x_raw[m,1]
            score_rows.append(dict(policy=policy, model="persistence", evaluation_pool=pool,
                                   scheduled_origins=int(scheduled_test.sum()), issued_origins=int(issue.sum()),
                                   prediction_coverage_pct=row["prediction_coverage_pct"],
                                   available_target_origins=int(eval_test.sum()),
                                   scorable_prediction_coverage_pct=row["scorable_prediction_coverage_pct"], **metrics(y[m], baseline_pred)))
        for month in pd.period_range("2026-01", "2026-06", freq="M"):
            month_mask = scheduled_test & (ix.to_period("M") == month)
            for pool, m in [("policy_issued", month_mask & target_ok & windows[policy]),
                            ("common_intersection", month_mask & common_test)]:
                month_rows.append(dict(month=str(month), policy=policy, evaluation_pool=pool,
                                       scheduled_origins=int(month_mask.sum()),
                                       issued_origins=int((month_mask & windows[policy]).sum()),
                                       prediction_coverage_pct=100*float((month_mask & windows[policy]).sum()/month_mask.sum()),
                                       **metrics(y[m], predictions[m])))
    for f in flags.columns:
        frame[f"history_{f}_count"] = flags[f].astype(int).rolling(WINDOW,min_periods=WINDOW).sum().to_numpy()[test_idx]
    frame["history_imputed_cells"] = imputed.astype(int).sum(axis=1).rolling(WINDOW,min_periods=WINDOW).sum().to_numpy()[test_idx]
    frame["history_max_imputation_age_min"] = ages.max(axis=1).rolling(WINDOW,min_periods=WINDOW).max().to_numpy()[test_idx]
    frame.to_csv(out / "test_predictions.csv.gz", index=False, compression=dict(method="gzip", mtime=0))
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / "test_policy_scores.csv", index=False)
    pd.DataFrame(month_rows).to_csv(out / "monthly_test_scores.csv", index=False)
    pd.DataFrame(ncounts).to_csv(out / "split_sample_counts.csv", index=False)
    boot = aggregate_bootstrap(frame, frame.common_evaluation_origin.to_numpy())
    boot.to_csv(out / "paired_block_bootstrap.csv", index=False)
    subsets = dict(
        common_intersection=frame.common_evaluation_origin,
        permissive_additional_to_strict=frame.issued_permissive & ~frame.issued_strict & frame.target_reference_usable,
        raw_excluded_by_permissive=frame.issued_raw & ~frame.issued_permissive & frame.target_reference_usable,
        imputation_additional_to_permissive=frame.issued_causal_imputation & ~frame.issued_permissive & frame.target_reference_usable)
    marginal = []
    for subset, m in subsets.items():
        for p in POLICIES:
            q = m & frame[f"prediction_{p}_mw"].notna()
            marginal.append(dict(subset=subset, policy=p, eligible_subset_n=int(m.sum()),
                                 **metrics(frame.loc[q,"target_demand_mw"].to_numpy(), frame.loc[q,f"prediction_{p}_mw"].to_numpy())))
    pd.DataFrame(marginal).to_csv(out / "marginal_test_scores.csv", index=False)
    common_rows = scores[(scores.model == "histogram_gradient_boosting") & (scores.evaluation_pool == "common_intersection")]
    issued_rows = scores[(scores.model == "histogram_gradient_boosting") & (scores.evaluation_pool == "policy_issued")].set_index("policy")
    table = common_rows[["policy","issued_origins","prediction_coverage_pct","available_target_origins","scorable_prediction_coverage_pct","n_scored","rmse_mw","mae_mw","mape_pct"]].copy()
    table = table.rename(columns={"n_scored":"common_n", "rmse_mw":"common_rmse_mw", "mae_mw":"common_mae_mw","mape_pct":"common_mape_pct"})
    table["issued_scored_n"] = table.policy.map(issued_rows.n_scored)
    table["issued_rmse_mw"] = table.policy.map(issued_rows.rmse_mw)
    table["issued_mae_mw"] = table.policy.map(issued_rows.mae_mw)
    table["issued_mape_pct"] = table.policy.map(issued_rows.mape_pct)
    table.to_csv(out / "main_table.csv", index=False)
    summary = dict(
        task="One-hour-ahead released national demand forecast from a one-hour seven-field history",
        source_hashes=hashes, observed_records=int(grid[FIELDS].notna().all(axis=1).sum()), expected_grid_points=int(len(ix)),
        feature_fields=FIELDS, feature_count=int(x_raw.shape[1]), lag_steps=list(range(12)), horizon_steps=HORIZON,
        lag_order="t, t-5 minutes, ..., t-55 minutes", calendar_features="Six sine/cosine terms for target hour, weekday, and annual position",
        response="Released demand at t+60 minutes minus demand input at t; add the same demand input to model output",
        training_origin_start=str(ix[WINDOW-1]), training_target_end_exclusive=str(TRAIN_END),
        validation_origin_start=str(TRAIN_END), validation_target_end_exclusive=str(TEST_START),
        refit_target_end_exclusive=str(TEST_START), test_origin_start=str(ix[scheduled_test].min()),
        test_origin_end=str(ix[scheduled_test].max()), scheduled_test_origins=int(scheduled_test.sum()),
        observed_reference_test_origins=int(eval_test.sum()), excluded_reference_test_origins=int((scheduled_test & ~target_ok).sum()),
        common_test_origins=int(common_test.sum()), common_validation_origins=int(common_val.sum()),
        target_screen="Observed positive released demand and absolute supply-identity residual <= 1 MW; no target imputation",
        source_availability="Retrospective quarterly archive; issue-time availability and revisions cannot be verified",
        raw_policy="All seven fields observed at every point in the 12-point history; observed flagged values retained unchanged",
        strict_policy="Full-service strict point gate at all 12 history points; any required-rule flag blocks issuance",
        permissive_policy="Full-service permissive gate at all 12 history points; nesting ambiguity retained with reason flags",
        imputation_policy="Field-specific masking followed by past-only last observation carried forward; nesting ambiguity retained and flagged",
        imputation_mask_mapping={"missing_timestamp":FIELDS,"forecast_zero":[FIELDS[2]],"supply_identity":FIELDS[:2]+FIELDS[3:5],
                                 "operating_zero_or_conflict_or_transition":FIELDS[5:],"nesting_ambiguity":[]},
        imputation_max_age_minutes=float(ages.max().max()), imputation_test_max_age_minutes=float(frame.history_max_imputation_age_min.max()),
        imputation_age_limit="No age limit; the maximum age is disclosed and reconstructed cells stay distinguishable",
        quality_flag_counts={c:int(flags[c].sum()) for c in flags},
        feature_flags="Reason flags and imputation provenance retained in predictions; not used as model features",
        hyperparameter_selection="One common iteration count minimizes mean RMSE across four policies on their common chronological validation origins",
        max_iter_candidates=CANDIDATES, selected_max_iter=choice, model_parameters=PARAMS,
        training_protocol="Each policy has its own eligible training pool; all share calendar periods, features, parameters, and seed; refit on all pre-test data",
        interpretation="Common-origin contrasts control evaluated timestamps, but also reflect policy-dependent training selection; no causal or operational benefit estimate",
        bootstrap="2000 paired moving-block draws of seven consecutive calendar days; fixed fitted models; percentile 95% intervals",
        no_future_feature_or_imputation=True, no_random_train_test_split=True, no_early_stopping_random_split=True,
        no_archived_forecast_vintage_claim=True, no_alarm_ground_truth_claim=True,
        software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__),
        threads=2, seed=SEED)
    (out / "run_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding="utf-8")
    summarize_flagged_episodes(out / "test_predictions.csv.gz", out)
    print(table.to_string(index=False), flush=True)
    print(f"Completed in {time.time()-started:.1f} seconds: {out}", flush=True)


if __name__ == "__main__":
    main()
