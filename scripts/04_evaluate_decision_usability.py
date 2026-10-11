from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import shutil
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived_data"
OUT = ROOT / "reproduced_results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
VALIDATION = ROOT / "validation"
for p in (TABLES, FIGURES, VALIDATION):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260830
N_RANDOM = 1000
rng = np.random.default_rng(SEED)

PANEL_PATH = DERIVED / "audited_panel.csv"
if not PANEL_PATH.exists():
    raise FileNotFoundError("Run 01_build_audited_panel.py first.")

obs = pd.read_csv(PANEL_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
START = pd.Timestamp("2024-07-01 00:00:00")
END = pd.Timestamp("2026-06-30 23:55:00")
expected_index = pd.date_range(START, END, freq="5min")
N = len(expected_index)

# Full expected grid. All timestamps are interpreted as local Korean Standard Time
# (UTC+09:00); South Korea does not use daylight-saving time in the study period.
grid = pd.DataFrame({"timestamp": expected_index})
grid = grid.merge(obs, on="timestamp", how="left", suffixes=("", "_obs"))
grid["observed"] = grid["source_file"].notna()

# ---------------------------------------------------------------------------
# 1. Evidence status and operational disposition
# ---------------------------------------------------------------------------
# Reason flags are kept separately. A primary evidence status is assigned only
# for compact occupancy reporting; the operational disposition is a distinct axis.
grid["flag_forecast_zero"] = grid["observed"] & grid["maximum_forecast_demand_mw"].eq(0)
grid["flag_supply_identity"] = grid["observed"] & grid["supply_identity_residual_mw"].abs().gt(1.0)
grid["flag_operating_zero"] = grid["observed"] & grid["operating_reserve_mw"].eq(0)
grid["flag_low_operating_high_supply"] = (
    grid["observed"]
    & grid["operating_reserve_mw"].lt(1500.0)
    & grid["supply_reserve_mw"].ge(5500.0)
)
grid["flag_nesting_ambiguity"] = grid["observed"] & grid["reserve_nesting_excess_mw"].gt(1e-9)

# The interval was selected after inspection of the archive.
# Sensitivity analyses vary its boundaries.
transition_start = pd.Timestamp("2026-06-30 16:20:00")
transition_end = pd.Timestamp("2026-06-30 17:10:00")
grid["flag_unresolved_transition"] = grid["observed"] & grid["timestamp"].between(transition_start, transition_end)

deterministic_failure = grid[["flag_forecast_zero", "flag_supply_identity", "flag_operating_zero"]].any(axis=1)
cross_field_conflict = grid["flag_low_operating_high_supply"] & ~deterministic_failure
unresolved_transition = grid["flag_unresolved_transition"] & ~deterministic_failure & ~cross_field_conflict
semantic_ambiguity = (
    grid["flag_nesting_ambiguity"]
    & ~deterministic_failure
    & ~cross_field_conflict
    & ~unresolved_transition
)

status = np.full(N, "passed_within_declared_coverage", dtype=object)
status[~grid["observed"].to_numpy()] = "missing_or_unresolved"
status[semantic_ambiguity.to_numpy()] = "semantic_ambiguity"
status[unresolved_transition.to_numpy()] = "unresolved_transition"
status[cross_field_conflict.to_numpy()] = "cross_field_conflict"
status[deterministic_failure.to_numpy()] = "deterministic_field_failure"
grid["primary_evidence_status"] = status

# Base operational disposition (permissive policy for semantic ambiguity).
disposition = np.full(N, "release", dtype=object)
disposition[semantic_ambiguity.to_numpy()] = "release_with_flag"
disposition[(unresolved_transition | cross_field_conflict).to_numpy()] = "review"
disposition[deterministic_failure.to_numpy()] = "block_or_quarantine"
disposition[~grid["observed"].to_numpy()] = "break_window"
grid["base_operational_disposition"] = disposition

status_order = [
    "passed_within_declared_coverage",
    "deterministic_field_failure",
    "cross_field_conflict",
    "semantic_ambiguity",
    "unresolved_transition",
    "missing_or_unresolved",
]
status_definitions = {
    "passed_within_declared_coverage": "Observed and no declared deterministic failure, conflict, ambiguity, or unresolved transition",
    "deterministic_field_failure": "Zero-valued required field or exact supply-identity failure",
    "cross_field_conflict": "Low operating reserve with high supply reserve, without a zero-valued field",
    "semantic_ambiguity": "Declared reserve-nesting check is flagged but public metadata do not resolve definition or synchronization",
    "unresolved_transition": "Observed point within the 30 June 2026 inspection context",
    "missing_or_unresolved": "Expected timestamp not released",
}
status_rows = []
for s in status_order:
    c = int((grid["primary_evidence_status"] == s).sum())
    status_rows.append({
        "evidence_status": s,
        "grid_points": c,
        "share_of_expected_grid_pct": 100 * c / N,
        "definition": status_definitions[s],
    })
status_table = pd.DataFrame(status_rows)
status_table.to_csv(TABLES / "evidence_status_occupancy.csv", index=False)

disp_order = ["release", "release_with_flag", "review", "block_or_quarantine", "break_window"]
disp_rows = []
for d in disp_order:
    c = int((grid["base_operational_disposition"] == d).sum())
    disp_rows.append({"operational_disposition": d, "grid_points": c, "share_of_expected_grid_pct": 100*c/N})
pd.DataFrame(disp_rows).to_csv(TABLES / "operational_disposition_occupancy.csv", index=False)

grid[[
    "timestamp", "observed", "primary_evidence_status", "base_operational_disposition",
    "flag_forecast_zero", "flag_supply_identity", "flag_operating_zero",
    "flag_low_operating_high_supply", "flag_nesting_ambiguity", "flag_unresolved_transition"
]].to_csv(TABLES / "quality_status_panel.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Mission-specific, state-aware reliability
# ---------------------------------------------------------------------------
# A mission is available at a grid point when all fields and evidence conditions
# required by that mission are usable under its stated ambiguity policy.
point_available: dict[str, pd.Series] = {}
point_available["cadence_delivery"] = grid["observed"].copy()
point_available["forecast_planning"] = grid["observed"] & ~grid["flag_forecast_zero"]

reserve_base = (
    grid["observed"]
    & ~grid["flag_supply_identity"]
    & ~grid["flag_operating_zero"]
    & ~grid["flag_low_operating_high_supply"]
    & ~grid["flag_unresolved_transition"]
)
point_available["reserve_stage_permissive"] = reserve_base
point_available["reserve_stage_strict"] = reserve_base & ~grid["flag_nesting_ambiguity"]

full_base = reserve_base & ~grid["flag_forecast_zero"]
point_available["full_service_permissive"] = full_base
point_available["full_service_strict"] = full_base & ~grid["flag_nesting_ambiguity"]

mission_meta = [
    ("cadence_delivery", "Cadence delivery", "Timestamp presence only", "All observed records"),
    ("forecast_planning", "Forecast planning", "Timestamp and positive maximum-demand forecast", "Other reserve-field flags do not invalidate this mission"),
    ("reserve_stage_permissive", "Reserve-stage monitoring (permissive)", "Timestamp, supply and operating reserve; deterministic failures, cross-field conflicts, and unresolved transition excluded", "Reserve-nesting ambiguity may be consumed only with an explicit flag"),
    ("reserve_stage_strict", "Reserve-stage monitoring (strict)", "Permissive requirements plus declared reserve-nesting check", "Semantic ambiguity breaks the mission window"),
    ("full_service_permissive", "Full-service use (permissive)", "Forecast and reserve-stage requirements", "Reserve-nesting ambiguity may be consumed only with an explicit flag"),
    ("full_service_strict", "Full-service use (strict)", "All required released fields and declared deterministic or definition-linked checks", "Only passed-within-coverage points satisfy the mission"),
]
pd.DataFrame(mission_meta, columns=["mission_code", "mission", "required_evidence", "ambiguity_policy"]).to_csv(TABLES / "mission_profiles.csv", index=False)

WINDOWS = [5, 10, 15, 30, 60]
mission_rows: list[dict] = []
mission_window_series: dict[tuple[str, int], pd.Series] = {}


def longest_true_run(series: pd.Series) -> tuple[int, pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    if not series.any():
        return 0, pd.NaT, pd.NaT
    groups = series.ne(series.shift()).cumsum()
    best = None
    for _, sub in series[series].groupby(groups[series]):
        item = (len(sub), sub.index.min(), sub.index.max())
        if best is None or item[0] > best[0]:
            best = item
    assert best is not None
    return int(best[0] * 5), pd.Timestamp(best[1]), pd.Timestamp(best[2])

for code, label, _, policy in mission_meta:
    p = point_available[code].astype(int)
    for minutes in WINDOWS:
        n = minutes // 5
        counts = p.rolling(n, min_periods=n).sum()
        eligible = counts.iloc[n-1:]
        available = eligible.eq(n)
        available.index = expected_index[n-1:]
        mission_window_series[(code, minutes)] = available
        unavailable = ~available
        longest_min, run_start, run_end = longest_true_run(unavailable)
        mission_rows.append({
            "mission_code": code,
            "mission": label,
            "ambiguity_policy": policy,
            "window_minutes": minutes,
            "eligible_decision_times": int(len(available)),
            "available_decision_times": int(available.sum()),
            "unavailable_decision_times": int(unavailable.sum()),
            "mission_reliability_pct": 100 * float(available.mean()),
            "mission_unavailability_pct": 100 * float(unavailable.mean()),
            "longest_unavailable_run_minutes": longest_min,
            "longest_run_start": run_start,
            "longest_run_end": run_end,
        })
mission_reliability = pd.DataFrame(mission_rows)
mission_reliability.to_csv(TABLES / "mission_specific_window_reliability.csv", index=False)

# Presence-only window availability.
cadence_compat = mission_reliability[mission_reliability["mission_code"].eq("cadence_delivery")].copy()
row_missing_share = 1 - float(grid["observed"].mean())
cadence_compat["row_missing_share_pct"] = 100 * row_missing_share
cadence_compat["missingness_amplification_factor"] = cadence_compat["mission_unavailability_pct"] / (100*row_missing_share)
cadence_compat.rename(columns={"mission_reliability_pct":"data_service_reliability_pct", "mission_unavailability_pct":"unavailable_share_pct"}).to_csv(TABLES / "decision_window_reliability.csv", index=False)

# Monthly and quarterly 60-minute reliability by mission.
period_rows = []
for code, label, _, policy in mission_meta:
    a = mission_window_series[(code, 60)]
    tmp = pd.DataFrame({"timestamp": a.index, "available": a.to_numpy(dtype=bool)})
    tmp["month"] = tmp["timestamp"].dt.to_period("M").astype(str)
    tmp["quarter"] = tmp["timestamp"].dt.to_period("Q").astype(str)
    for period_type in ["month", "quarter"]:
        grp = tmp.groupby(period_type)["available"].agg(["count","sum","mean"]).reset_index()
        for _, r in grp.iterrows():
            period_rows.append({
                "period_type": period_type,
                "period": r[period_type],
                "mission_code": code,
                "mission": label,
                "ambiguity_policy": policy,
                "eligible_decision_times": int(r["count"]),
                "available_decision_times": int(r["sum"]),
                "mission_reliability_pct": 100*float(r["mean"]),
            })
period_reliability = pd.DataFrame(period_rows)
period_reliability[period_reliability.period_type.eq("month")].to_csv(TABLES / "monthly_mission_reliability_60min.csv", index=False)
period_reliability[period_reliability.period_type.eq("quarter")].to_csv(TABLES / "quarterly_mission_reliability_60min.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Missing-placement benchmarks and episode geometry
# ---------------------------------------------------------------------------
missing_positions = np.flatnonzero(~grid["observed"].to_numpy())
m = len(missing_positions)


def union_window_stats(missing_pos: np.ndarray, n: int, total_n: int) -> tuple[int, int]:
    """Number and longest run of unavailable eligible windows.

    A window ends at e in [n-1, total_n-1]. A missing grid position j affects
    end times [max(n-1,j), min(total_n-1,j+n-1)].
    """
    if len(missing_pos) == 0:
        return 0, 0
    starts = np.maximum(n-1, missing_pos)
    ends = np.minimum(total_n-1, missing_pos + n - 1)
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    total = 0
    longest = 0
    cur_s, cur_e = int(starts[0]), int(ends[0])
    for s, e in zip(starts[1:], ends[1:]):
        s, e = int(s), int(e)
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            length = cur_e - cur_s + 1
            total += length
            longest = max(longest, length)
            cur_s, cur_e = s, e
    length = cur_e - cur_s + 1
    total += length
    longest = max(longest, length)
    return total, longest

# Observed missing episodes.
miss_bool = ~grid["observed"].to_numpy()
change = np.r_[True, miss_bool[1:] != miss_bool[:-1]]
run_starts = np.flatnonzero(change)
run_ends = np.r_[run_starts[1:]-1, N-1]
episode_lengths = np.array([e-s+1 for s,e in zip(run_starts,run_ends) if miss_bool[s]], dtype=int)


def uniform_composition(total: int, parts: int, gen: np.random.Generator) -> np.ndarray:
    if parts == 1:
        return np.array([total], dtype=int)
    # Stars-and-bars: choose separator positions uniformly.
    cuts = np.sort(gen.choice(np.arange(1, total + parts), size=parts-1, replace=False))
    ext = np.r_[0, cuts, total + parts]
    return np.diff(ext) - 1


def episode_preserving_positions(lengths: np.ndarray, total_n: int, gen: np.random.Generator) -> np.ndarray:
    lens = gen.permutation(lengths)
    e = len(lens)
    required_internal = e - 1
    slack = total_n - int(lens.sum()) - required_internal
    if slack < 0:
        raise ValueError("Episodes do not fit in grid")
    gaps_extra = uniform_composition(slack, e+1, gen)
    positions = []
    pos = int(gaps_extra[0])
    for i, L in enumerate(lens):
        positions.extend(range(pos, pos + int(L)))
        pos += int(L)
        if i < e-1:
            pos += 1 + int(gaps_extra[i+1])
    return np.asarray(positions, dtype=int)

benchmark_rows = []
random_summaries = {}
for minutes in WINDOWS:
    n = minutes // 5
    eligible = N - n + 1
    obs_count, obs_longest = union_window_stats(missing_positions, n, N)
    obs_share = obs_count / eligible
    p_complete = 1.0
    for i in range(n):
        p_complete *= (N - m - i) / (N - i)
    fixed_expect = 1 - p_complete

    rand_counts = np.empty(N_RANDOM, dtype=int)
    rand_longest = np.empty(N_RANDOM, dtype=int)
    ep_counts = np.empty(N_RANDOM, dtype=int)
    ep_longest = np.empty(N_RANDOM, dtype=int)
    for b in range(N_RANDOM):
        pos = np.sort(rng.choice(N, size=m, replace=False))
        rand_counts[b], rand_longest[b] = union_window_stats(pos, n, N)
        epos = episode_preserving_positions(episode_lengths, N, rng)
        ep_counts[b], ep_longest[b] = union_window_stats(epos, n, N)

    dispersed = np.rint(np.linspace(0, N-1, m)).astype(int)
    dispersed = np.unique(dispersed)
    # With N/m far larger than all n, linspace gives m unique positions.
    if len(dispersed) != m:
        raise RuntimeError("Dispersed reference did not produce unique positions")
    disp_count, disp_longest = union_window_stats(dispersed, n, N)
    cluster_start = (N-m)//2
    concentrated = np.arange(cluster_start, cluster_start+m, dtype=int)
    conc_count, conc_longest = union_window_stats(concentrated, n, N)

    random_summaries[minutes] = {
        "random_counts": rand_counts,
        "episode_counts": ep_counts,
        "random_longest": rand_longest,
        "episode_longest": ep_longest,
    }
    benchmark_rows.append({
        "window_minutes": minutes,
        "eligible_decision_times": eligible,
        "observed_unavailable_decisions": obs_count,
        "observed_unavailability_pct": 100*obs_share,
        "observed_longest_run_minutes": 5*obs_longest,
        "fixed_count_random_expected_unavailability_pct": 100*fixed_expect,
        "random_placement_mean_unavailability_pct": 100*rand_counts.mean()/eligible,
        "random_placement_lower_95_pct": 100*np.quantile(rand_counts/eligible, 0.025),
        "random_placement_upper_95_pct": 100*np.quantile(rand_counts/eligible, 0.975),
        "random_placement_median_longest_run_minutes": 5*float(np.median(rand_longest)),
        "episode_preserving_mean_unavailability_pct": 100*ep_counts.mean()/eligible,
        "episode_preserving_lower_95_pct": 100*np.quantile(ep_counts/eligible, 0.025),
        "episode_preserving_upper_95_pct": 100*np.quantile(ep_counts/eligible, 0.975),
        "episode_preserving_median_longest_run_minutes": 5*float(np.median(ep_longest)),
        "widely_dispersed_unavailability_pct": 100*disp_count/eligible,
        "widely_dispersed_longest_run_minutes": 5*disp_longest,
        "single_cluster_unavailability_pct": 100*conc_count/eligible,
        "single_cluster_longest_run_minutes": 5*conc_longest,
        "observed_to_fixed_random_ratio": obs_share/fixed_expect if fixed_expect else np.nan,
    })
benchmarks = pd.DataFrame(benchmark_rows)
benchmarks.to_csv(TABLES / "missingness_placement_benchmarks.csv", index=False)

# Missing-episode start intervals for descriptive rare-event monitoring.
episode_starts = np.array([s for s in run_starts if miss_bool[s]], dtype=int)
intervals = np.diff(np.r_[-1, episode_starts])
p_hat = len(episode_starts)/N
center = 1/p_hat
q01 = int(math.ceil(math.log(1-0.01)/math.log(1-p_hat)))
q99 = int(math.ceil(math.log(1-0.99)/math.log(1-p_hat)))
monitor = pd.DataFrame({
    "episode_id": np.arange(1, len(episode_starts)+1),
    "episode_start": expected_index[episode_starts],
    "five_minute_opportunities_since_prior_start": intervals,
    "minutes_since_prior_start": 5*intervals,
    "geometric_center_opportunities": center,
    "geometric_1pct_limit_opportunities": q01,
    "geometric_99pct_limit_opportunities": q99,
})
monitor.to_csv(TABLES / "time_between_missing_episode_starts.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Sensitivity analyses
# ---------------------------------------------------------------------------
# Supply-identity tolerance.
tolerance_rows = []
resid_units = np.rint(obs["supply_identity_residual_mw"].abs() * 10000).astype(np.int64)
resid = resid_units / 10000
for tol in [0.1, 0.2, 0.5, 0.51, 0.6, 1, 5, 10, 100, 1000]:
    tolerance_rows.append({
        "tolerance_mw": tol,
        "records_exceeding_tolerance": int((resid_units > round(tol * 10000)).sum()),
        "minimum_residual_above_tolerance_mw": float(resid[resid_units > round(tol * 10000)].min()) if (resid_units > round(tol * 10000)).any() else np.nan,
        "maximum_residual_at_or_below_tolerance_mw": float(resid[resid_units <= round(tol * 10000)].max()) if (resid_units <= round(tol * 10000)).any() else np.nan,
    })
pd.DataFrame(tolerance_rows).to_csv(TABLES / "supply_identity_tolerance_sensitivity.csv", index=False)

# Transition-boundary sensitivity for strict full-service 60-minute reliability.
transition_scenarios = [
    ("none", None, None),
    ("extreme_change_core", pd.Timestamp("2026-06-30 16:35:00"), pd.Timestamp("2026-06-30 16:40:00")),
    ("contextual_11_points", transition_start, transition_end),
    ("expanded_20_minutes_each_side", pd.Timestamp("2026-06-30 16:00:00"), pd.Timestamp("2026-06-30 17:30:00")),
]
transition_rows=[]
for name, s, e in transition_scenarios:
    trans = pd.Series(False, index=grid.index)
    if s is not None:
        trans = grid["observed"] & grid["timestamp"].between(s,e)
    accept = (
        grid["observed"]
        & ~grid["flag_forecast_zero"]
        & ~grid["flag_supply_identity"]
        & ~grid["flag_operating_zero"]
        & ~grid["flag_low_operating_high_supply"]
        & ~grid["flag_nesting_ambiguity"]
        & ~trans
    )
    n=12
    a=accept.astype(int).rolling(n,min_periods=n).sum().iloc[n-1:].eq(n)
    a.index=expected_index[n-1:]
    longest, rs, re = longest_true_run(~a)
    transition_rows.append({
        "scenario": name,
        "transition_start": s,
        "transition_end": e,
        "observed_points_designated_unresolved": int(trans.sum()),
        "strict_full_service_60min_reliability_pct": 100*float(a.mean()),
        "strict_full_service_60min_unavailable_decisions": int((~a).sum()),
        "longest_unavailable_run_minutes": longest,
        "longest_run_start": rs,
        "longest_run_end": re,
    })
pd.DataFrame(transition_rows).to_csv(TABLES / "transition_window_sensitivity.csv", index=False)

# Ambiguity-policy comparison at 60 minutes.
policy_rows=[]
for code in ["reserve_stage_permissive","reserve_stage_strict","full_service_permissive","full_service_strict"]:
    r=mission_reliability[(mission_reliability.mission_code==code)&(mission_reliability.window_minutes==60)].iloc[0]
    policy_rows.append(r.to_dict())
pd.DataFrame(policy_rows).to_csv(TABLES / "ambiguity_policy_sensitivity_60min.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Predicate-level implementation tests (not detection-performance validation)
# ---------------------------------------------------------------------------
def point_predicates(record: dict[str,float]) -> set[str]:
    flags=set()
    if float(record["maximum_forecast_demand_mw"]) <= 0: flags.add("forecast_domain")
    expected=float(record["supply_capacity_mw"])-float(record["current_demand_mw"])
    if abs(float(record["supply_reserve_mw"])-expected)>1.0: flags.add("supply_identity")
    if float(record["operating_reserve_mw"])==0: flags.add("operating_zero")
    if float(record["operating_reserve_mw"])<1500 and float(record["supply_reserve_mw"])>=5500: flags.add("low_operating_high_supply")
    if float(record["operating_reserve_mw"])>float(record["supply_reserve_mw"]): flags.add("reserve_nesting")
    return flags

clean_obs = obs[
    obs["maximum_forecast_demand_mw"].gt(0)
    & obs["supply_identity_residual_mw"].abs().le(1)
    & obs["operating_reserve_mw"].gt(1500)
    & obs["reserve_nesting_excess_mw"].le(0)
].iloc[1000].copy()
base={k:float(clean_obs[k]) for k in ["supply_capacity_mw","current_demand_mw","maximum_forecast_demand_mw","supply_reserve_mw","operating_reserve_mw"]}

tests=[]
def add_test(name, changed, expected_flags, note):
    flags=point_predicates(changed)
    tests.append({
        "test_case":name,
        "expected_flags":";".join(sorted(expected_flags)) or "none",
        "observed_flags":";".join(sorted(flags)) or "none",
        "test_passed": flags==set(expected_flags),
        "interpretation":note,
    })

r=base.copy(); r["maximum_forecast_demand_mw"]=0; add_test("forecast_zero",r,{"forecast_domain"},"Domain predicate routes a zero-valued planning value")
for off, expected in [(0.5,set()),(1.0,set()),(1.01,{"supply_identity"}),(10,{"supply_identity"}),(1000,{"supply_identity"})]:
    r=base.copy(); r["supply_reserve_mw"]=(r["supply_capacity_mw"]-r["current_demand_mw"])+off
    add_test(f"identity_offset_{off:g}_mw",r,expected,"Boundary test for the stated 1 MW tolerance")
for v, expected in [(1499,{"low_operating_high_supply"}),(1500,set()),(1501,set())]:
    r=base.copy(); r["operating_reserve_mw"]=v
    add_test(f"operating_reserve_{v}_mw",r,expected,"Boundary test for the severe-stage review screen")
for ex, expected in [(-0.1,set()),(0,set()),(0.1,{"reserve_nesting"}),(100,{"reserve_nesting"})]:
    r=base.copy(); r["operating_reserve_mw"]=r["supply_reserve_mw"]+ex
    add_test(f"nesting_excess_{ex:g}_mw",r,expected,"Boundary test for the declared reserve-nesting check")
r=base.copy(); r["operating_reserve_mw"]=0; add_test("operating_zero",r,{"operating_zero","low_operating_high_supply"},"Reason flags retain both zero values and cross-field conflict")
r={k:v*1.01 for k,v in base.items()}; add_test("coherent_one_percent_scaling",r,set(),"Negative test: point-level deterministic invariants are preserved")
impl=pd.DataFrame(tests)
impl.to_csv(TABLES / "predicate_implementation_tests.csv",index=False)

# ---------------------------------------------------------------------------
# 6. Source-file manifest
# ---------------------------------------------------------------------------
manifest=[]
for p in sorted((ROOT/"data_raw").glob("*.csv")):
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    sub=obs[obs["source_file"].eq(p.name)]
    manifest.append({
        "file":p.name,
        "sha256":h,
        "first_timestamp_kst":sub["timestamp"].min(),
        "last_timestamp_kst":sub["timestamp"].max(),
        "observed_rows":len(sub),
        "download_date_encoded_in_source":"not encoded",
        "time_zone":"Korea Standard Time (UTC+09:00)",
    })
pd.DataFrame(manifest).to_csv(TABLES / "source_file_manifest_sha256.csv",index=False)

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
# Figure 1: quality-evidence and consumer-policy workflow.
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")
boxes = [
    (0.01, 0.42, 0.174, 0.50, "Coverage declaration\nField-by-rule\nevidence basis"),
    (0.21, 0.42, 0.174, 0.50, "Evidence status\nPass, failure, conflict,\nambiguity, unresolved"),
    (0.41, 0.42, 0.174, 0.50, "Operational disposition\nRelease, flag, review,\nblock, break window"),
    (0.61, 0.42, 0.174, 0.50, "Mission policy\nRequired fields and\nacceptable states"),
    (0.81, 0.42, 0.174, 0.50, "Window reliability\nUsable-window share,\nlongest interruption"),
]
for x, y, w, h, text in boxes:
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.2, transform=ax.transAxes))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10.2, transform=ax.transAxes)
for i in range(len(boxes)-1):
    x, y, w, h, _ = boxes[i]
    nx, ny, nw, nh, _ = boxes[i+1]
    ax.annotate("", xy=(nx, ny+nh/2), xytext=(x+w, y+h/2), xycoords=ax.transAxes,
                textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.1))
ax.text(0.5, 0.25, "Reason flags accompany each primary status.", ha="center", fontsize=11,
        transform=ax.transAxes)
ax.text(0.5, 0.09, "Consumer policies specify the required fields, evidence states, and observation windows.",
        ha="center", fontsize=10.5, transform=ax.transAxes)
fig.tight_layout()
fig.savefig(FIGURES/"Figure_1_state_aware_reliability_workflow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 3: 60-minute mission reliabilities.
main_codes=["cadence_delivery","forecast_planning","reserve_stage_permissive","reserve_stage_strict","full_service_permissive","full_service_strict"]
sub=mission_reliability[(mission_reliability.window_minutes==60)&mission_reliability.mission_code.isin(main_codes)].copy()
sub["short_label"]=["Cadence","Forecast","Reserve\npermissive","Reserve\nstrict","Full service\npermissive","Full service\nstrict"]
fig,ax=plt.subplots(figsize=(9,5.3))
ax.bar(sub["short_label"],sub["mission_reliability_pct"])
ax.set_ylabel("60-minute mission reliability (%)")
ax.set_ylim(min(93.5,sub["mission_reliability_pct"].min()-0.5),100)
for i,v in enumerate(sub["mission_reliability_pct"]): ax.text(i,v+0.08,f"{v:.3f}%",ha="center",fontsize=8)
ax.tick_params(axis="x",labelsize=8)
fig.tight_layout()
fig.savefig(FIGURES/"Figure_3_mission_specific_reliability.png",dpi=300,bbox_inches="tight")
plt.close(fig)

# Figure 4: observed and placement benchmarks.
fig,ax=plt.subplots(figsize=(8.8,5.5))
x=benchmarks["window_minutes"]
ax.plot(x,benchmarks["observed_unavailability_pct"],marker="o",label="Observed")
ax.plot(x,benchmarks["fixed_count_random_expected_unavailability_pct"],marker="s",label="Fixed-count random expectation")
ax.fill_between(x,benchmarks["random_placement_lower_95_pct"],benchmarks["random_placement_upper_95_pct"],alpha=0.2,label="Random-placement 95% interval")
ax.plot(x,benchmarks["episode_preserving_mean_unavailability_pct"],marker="^",label="Episode-preserving mean")
ax.plot(x,benchmarks["widely_dispersed_unavailability_pct"],marker="d",label="Widely dispersed")
ax.set_xlabel("Required trailing window (minutes)")
ax.set_ylabel("Unavailable decision times (%)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES/"Figure_4_missingness_placement_benchmarks.png",dpi=300,bbox_inches="tight")
plt.close(fig)

# Figure 5: monthly 60-minute reliability, selected missions.
monthly=period_reliability[period_reliability.period_type.eq("month")].copy()
fig,ax=plt.subplots(figsize=(10.5,5.6))
for code,label in [("cadence_delivery","Cadence"),("forecast_planning","Forecast"),("reserve_stage_strict","Reserve strict"),("full_service_strict","Full service strict")]:
    s=monthly[monthly.mission_code.eq(code)].copy()
    ax.plot(pd.to_datetime(s.period),s.mission_reliability_pct,marker="o",markersize=3,label=label)
ax.set_ylabel("60-minute mission reliability (%)")
ax.set_xlabel("Decision month")
ax.legend(fontsize=8,ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIGURES/"Figure_5_monthly_mission_reliability.png",dpi=300,bbox_inches="tight")
plt.close(fig)

# Supplementary time-between-event chart.
fig,ax=plt.subplots(figsize=(10,5.2))
ax.plot(monitor["episode_id"],monitor["five_minute_opportunities_since_prior_start"],linewidth=0.8)
ax.axhline(center,linestyle="--",label=f"Geometric center = {center:.1f}")
ax.axhline(q99,linestyle=":",label=f"99% upper reference = {q99}")
ax.set_xlabel("Missing-episode sequence")
ax.set_ylabel("Five-minute opportunities since prior episode start")
ax.set_yscale("log")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES/"Supplementary_Figure_S5_time_between_missing_episodes.png",dpi=300,bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Key-results file
# ---------------------------------------------------------------------------
def metric(code,window=60):
    r=mission_reliability[(mission_reliability.mission_code==code)&(mission_reliability.window_minutes==window)].iloc[0]
    return float(r.mission_reliability_pct)

b60=benchmarks[benchmarks.window_minutes.eq(60)].iloc[0]
key={
    "seed":SEED,
    "randomization_repetitions":N_RANDOM,
    "expected_grid_points":N,
    "observed_records":int(grid.observed.sum()),
    "missing_timestamps":m,
    "row_missing_share_pct":100*row_missing_share,
    "cadence_60min_reliability_pct":metric("cadence_delivery"),
    "forecast_60min_reliability_pct":metric("forecast_planning"),
    "reserve_permissive_60min_reliability_pct":metric("reserve_stage_permissive"),
    "reserve_strict_60min_reliability_pct":metric("reserve_stage_strict"),
    "full_service_permissive_60min_reliability_pct":metric("full_service_permissive"),
    "full_service_strict_60min_reliability_pct":metric("full_service_strict"),
    "cadence_60min_observed_unavailability_pct":float(b60.observed_unavailability_pct),
    "cadence_60min_fixed_random_expected_unavailability_pct":float(b60.fixed_count_random_expected_unavailability_pct),
    "cadence_60min_observed_to_random_ratio":float(b60.observed_to_fixed_random_ratio),
    "cadence_60min_longest_run_minutes":int(b60.observed_longest_run_minutes),
    "evidence_passed_count":int((grid.primary_evidence_status=="passed_within_declared_coverage").sum()),
    "evidence_deterministic_failure_count":int((grid.primary_evidence_status=="deterministic_field_failure").sum()),
    "evidence_cross_field_conflict_count":int((grid.primary_evidence_status=="cross_field_conflict").sum()),
    "evidence_semantic_ambiguity_count":int((grid.primary_evidence_status=="semantic_ambiguity").sum()),
    "evidence_unresolved_transition_count":int((grid.primary_evidence_status=="unresolved_transition").sum()),
    "evidence_missing_count":int((grid.primary_evidence_status=="missing_or_unresolved").sum()),
    "predicate_tests":int(len(impl)),
    "predicate_tests_passed":int(impl.test_passed.sum()),
    "missing_episode_count":int(len(episode_lengths)),
    "time_between_episode_center_opportunities":float(center),
    "time_between_episode_99pct_limit_opportunities":int(q99),
}
(OUT/"mission_key_results.json").write_text(json.dumps(key,indent=2,default=str),encoding="utf-8")
print(json.dumps(key,indent=2))
print("\n60-minute mission reliability:\n",mission_reliability[mission_reliability.window_minutes.eq(60)][["mission","mission_reliability_pct","longest_unavailable_run_minutes"]].to_string(index=False))
print("\nMissingness placement benchmarks:\n",benchmarks.to_string(index=False))
print("\nEvidence status occupancy:\n",status_table.to_string(index=False))
print("\nPredicate implementation tests:",int(impl.test_passed.sum()),"/",len(impl),"passed")
