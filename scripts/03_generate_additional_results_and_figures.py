from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived_data"
OUT = ROOT / "reproduced_results"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def save_publication_figure(fig, stem):
    fig.savefig(FIGS / (stem + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGS / (stem + ".pdf"), bbox_inches="tight", facecolor="white",
                metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
    fig.savefig(FIGS / (stem + ".eps"), bbox_inches="tight", facecolor="white")

def clock_axis(ax, date, interval=10):
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, interval)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel(date + " | Time (KST; UTC+09:00)")
    ax.tick_params(axis="x", rotation=0)

df = pd.read_csv(DERIVED / "audited_panel.csv", parse_dates=["timestamp"])

# ---------- source inventory and missingness ----------
raw_dir = ROOT / "data_raw"
raw_files = sorted(raw_dir.glob("sukub_*.csv"))
inventory = []
all_expected = []
for path in raw_files:
    raw = pd.read_csv(path, encoding="cp949")
    ts = pd.to_datetime(raw["기준일시"].astype(str), format="%Y%m%d%H%M%S")
    stem = path.stem.replace("sukub_", "")
    start = pd.to_datetime(stem.split("-")[0], format="%Y%m%d")
    end = pd.to_datetime(stem.split("-")[1], format="%Y%m%d") + pd.Timedelta(hours=23, minutes=55)
    expected = pd.date_range(start, end, freq="5min")
    all_expected.append(expected)
    duplicates = int(ts.duplicated().sum())
    reverse = int((ts.diff().dt.total_seconds() < 0).sum())
    observed = len(raw)
    missing = len(expected) - observed
    days = (end.normalize() - start.normalize()).days + 1
    inventory.append({
        "file": path.name,
        "period_start": start,
        "period_end": end,
        "observed_rows": observed,
        "expected_rows": len(expected),
        "missing_rows": missing,
        "missing_per_day": missing / days,
        "duplicate_timestamps": duplicates,
        "reverse_timestamp_steps": reverse,
        "period": f"{start.year}Q{((start.month - 1) // 3) + 1}",
    })
inv = pd.DataFrame(inventory)
inv.to_csv(TABLES / "source_file_inventory.csv", index=False)
inv.to_csv(TABLES / "quarterly_missingness.csv", index=False)

expected_index = pd.DatetimeIndex(np.concatenate([x.values for x in all_expected]))
observed_index = pd.DatetimeIndex(df["timestamp"])
missing_index = expected_index.difference(observed_index).sort_values()
pd.DataFrame({"timestamp": missing_index}).to_csv(TABLES / "missing_timestamps.csv", index=False)

# Missing episodes
miss = pd.DataFrame({"timestamp": missing_index})
miss["new_episode"] = miss["timestamp"].diff().ne(pd.Timedelta(minutes=5)).cumsum()
miss_rows = []
for episode_id, sub in miss.groupby("new_episode"):
    miss_rows.append({
        "episode_id": int(episode_id),
        "start": sub["timestamp"].min(),
        "end": sub["timestamp"].max(),
        "missing_points": len(sub),
        "duration_minutes": len(sub) * 5,
    })
missing_episodes = pd.DataFrame(miss_rows)
missing_episodes.to_csv(TABLES / "missing_episodes.csv", index=False)

# Period trend
trend_rows = []
period_specs = [
    ("2024 H2", "2024-07-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026 H1", "2026-01-01", "2026-06-30"),
    ("2026 Q2", "2026-04-01", "2026-06-30"),
]
for label, start, end in period_specs:
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    count = int(((missing_index >= start) & (missing_index < end + pd.Timedelta(days=1))).sum())
    days = (end - start).days + 1
    trend_rows.append({"period": label, "missing_points": count, "days": days, "missing_per_day": count / days})
pd.DataFrame(trend_rows).to_csv(TABLES / "missingness_period_trend.csv", index=False)

# Domain summary
fields = [
    "supply_capacity_mw", "current_demand_mw", "maximum_forecast_demand_mw",
    "supply_reserve_mw", "supply_reserve_rate_pct",
    "operating_reserve_mw", "operating_reserve_rate_pct",
]
domain_rows = []
for field in fields:
    domain_rows.append({
        "field": field,
        "nulls": int(df[field].isna().sum()),
        "zeros": int(df[field].eq(0).sum()),
        "negatives": int(df[field].lt(0).sum()),
        "minimum": df[field].min(),
        "maximum": df[field].max(),
    })
pd.DataFrame(domain_rows).to_csv(TABLES / "field_domain_summary.csv", index=False)

# ---------- exact algebraic failures ----------
supply_fail = df[df["supply_identity_residual_mw"].abs() > 1].copy()
supply_fail.to_csv(TABLES / "supply_identity_failures.csv", index=False)

# ---------- severe candidates and signatures ----------
low = df[df["operating_reserve_mw"] < 1500].sort_values("timestamp").copy()
low["episode"] = low["timestamp"].diff().ne(pd.Timedelta(minutes=5)).cumsum()
segments = []
for segment_id, sub in low.groupby("episode"):
    first_idx = sub.index.min(); last_idx = sub.index.max()
    prev = df.loc[first_idx - 1] if first_idx > 0 else None
    nxt = df.loc[last_idx + 1] if last_idx + 1 < len(df) else None
    vals = sub["operating_reserve_mw"]
    if vals.eq(0).all():
        signature = "Zero-valued run"
    elif len(sub) >= 2 and vals.max() - vals.min() <= 1:
        signature = "Nearly constant nonzero field"
    else:
        signature = "Isolated low positive value"
    entry_jump = abs(vals.iloc[0] - prev["operating_reserve_mw"]) if prev is not None else np.nan
    exit_jump = abs(nxt["operating_reserve_mw"] - vals.iloc[-1]) if nxt is not None else np.nan
    segments.append({
        "segment_id": int(segment_id),
        "start": sub["timestamp"].min(),
        "end": sub["timestamp"].max(),
        "records": len(sub),
        "duration_minutes": len(sub) * 5,
        "failure_signature": signature,
        "minimum_operating_reserve_mw": vals.min(),
        "maximum_operating_reserve_mw": vals.max(),
        "minimum_supply_reserve_mw": sub["supply_reserve_mw"].min(),
        "maximum_supply_reserve_mw": sub["supply_reserve_mw"].max(),
        "operating_reserve_range_mw": vals.max() - vals.min(),
        "demand_range_mw": sub["current_demand_mw"].max() - sub["current_demand_mw"].min(),
        "entry_jump_mw": entry_jump,
        "exit_jump_mw": exit_jump,
        "screen_high_supply": bool(sub["supply_reserve_mw"].min() >= 5500),
        "boundary_values_normal": bool(prev is not None and nxt is not None and prev["operating_reserve_mw"] >= 4500 and nxt["operating_reserve_mw"] >= 4500),
        "large_boundary_jump": bool(entry_jump >= 5000 and exit_jump >= 5000),
    })
segments = pd.DataFrame(segments)
segments.to_csv(TABLES / "severe_candidate_signatures.csv", index=False)

# Gate contribution
pd.DataFrame([
    {"step":"Candidate severe-stage parsing","segments_remaining":3,"marginal_segments_removed":0,"role":"Defines records below 1,500 MW"},
    {"step":"High-supply screening: supply reserve >= 5,500 MW","segments_remaining":0,"marginal_segments_removed":3,"role":"Routes all three candidate segments to review in this archive"},
    {"step":"Normal adjacent operating-reserve values","segments_remaining":0,"marginal_segments_removed":0,"role":"Corroborates return to adjacent higher values"},
    {"step":"Boundary change >= 5,000 MW","segments_remaining":0,"marginal_segments_removed":0,"role":"Corroborates the observed discontinuity"},
    {"step":"Signature attribution","segments_remaining":0,"marginal_segments_removed":0,"role":"Distinguishes released-data signatures: nearly constant nonzero values, a zero-valued run, and an isolated low nonzero value"},
]).to_csv(TABLES / "gate_condition_contribution.csv", index=False)
pd.DataFrame([
    {"condition":"Cross-field high supply reserve (supply reserve >= 5,500 MW)","candidate_segments_quarantined":3,"incremental_segments_beyond_prior_conditions":3},
    {"condition":"Normal boundary operating reserve values","candidate_segments_quarantined":3,"incremental_segments_beyond_prior_conditions":0},
    {"condition":"Large boundary jump (>=5,000 MW)","candidate_segments_quarantined":3,"incremental_segments_beyond_prior_conditions":0},
]).to_csv(TABLES / "gate_condition_incremental_contribution.csv", index=False)

# Released-data signature examples
freeze = pd.DataFrame([
    {"field":"supply_reserve_mw","start":"2025-06-17 14:45:00","end":"2025-06-17 14:55:00","records":3,"signature":"Nearly constant nonzero subrun within algebraic failure","field_range":1.2,"independent_value_range":371.91},
    {"field":"operating_reserve_mw","start":"2025-02-27 15:05:00","end":"2025-02-27 15:10:00","records":2,"signature":"Nearly constant nonzero field","field_range":0.04,"independent_value_range":5197.1},
    {"field":"operating_reserve_mw","start":"2025-11-17 16:30:00","end":"2025-11-17 16:55:00","records":6,"signature":"Zero-valued run","field_range":0.0,"independent_value_range":813.2},
    {"field":"operating_reserve_mw","start":"2026-01-16 14:45:00","end":"2026-01-16 14:45:00","records":1,"signature":"Isolated low positive value","field_range":0.0,"independent_value_range":0.0},
])
freeze.to_csv(TABLES / "freeze_signature_examples.csv", index=False)

# ---------- persistence filtering ----------
low_full = df.set_index("timestamp")["operating_reserve_mw"].lt(1500)
filter_rows = []
for n in [1, 2, 3, 6]:
    # rolling only on contiguous observed rows; group by discontinuity
    group = df["dt_min"].ne(5).cumsum()
    flagged = pd.Series(False, index=df.index)
    for _, idx in df.groupby(group).groups.items():
        vals = df.loc[idx, "operating_reserve_mw"].lt(1500)
        flagged.loc[idx] = vals.rolling(n, min_periods=n).sum().eq(n).values
    # count original low episodes that contain at least one flagged decision
    surviving = 0
    for _, sub in low.groupby("episode"):
        if flagged.loc[sub.index].any(): surviving += 1
    filter_rows.append({
        "method": f"{n} consecutive low observation" + ("s" if n != 1 else ""),
        "nominal_persistence_minutes": n * 5,
        "segments_surviving": surviving,
        "decision_timestamps_flagged": int(flagged.sum()),
        "flagged_decision_minutes": int(flagged.sum()) * 5,
    })
filter_rows.append({
    "method":"Integrity screening plus signature attribution",
    "nominal_persistence_minutes":np.nan,
    "segments_surviving":0,
    "decision_timestamps_flagged":0,
    "flagged_decision_minutes":0,
})
pd.DataFrame(filter_rows).to_csv(TABLES / "persistence_filter_results.csv", index=False)

# ---------- decision availability ----------
presence = pd.Series(False, index=expected_index)
presence.loc[observed_index] = True
availability_rows = []
blind_runs = []
for minutes in [5,10,15,30,60]:
    n = minutes // 5
    roll = presence.astype(int).rolling(n, min_periods=n).sum()
    eligible = roll.iloc[n-1:]
    available = eligible.eq(n)
    unavailable = ~available
    # runs of unavailable decisions
    run_id = unavailable.ne(unavailable.shift()).cumsum()
    runs = []
    for _, sub in unavailable[unavailable].groupby(run_id[unavailable]):
        start = sub.index.min(); end = sub.index.max(); pts = len(sub)
        runs.append((start,end,pts,pts*5))
        blind_runs.append({"start":start,"end":end,"decision_points":pts,"duration_minutes":pts*5,"persistence_minutes":minutes})
    longest = max(runs, key=lambda x:x[3])
    availability_rows.append({
        "persistence_minutes":minutes,
        "eligible_decision_times":len(eligible),
        "available_decision_times":int(available.sum()),
        "unavailable_decision_times":int(unavailable.sum()),
        "unavailable_share_pct":100*unavailable.mean(),
        "longest_unavailable_run_minutes":longest[3],
        "longest_run_start":longest[0],
        "longest_run_end":longest[1],
    })
pd.DataFrame(availability_rows).to_csv(TABLES / "persistence_decision_availability.csv", index=False)
pd.DataFrame(blind_runs).to_csv(TABLES / "blind_window_runs.csv", index=False)

# Longest blind-window missing structure
long_date = pd.Timestamp("2025-05-29")
long_sub = missing_episodes[(pd.to_datetime(missing_episodes["start"]).dt.normalize() == long_date)]
long_sub = long_sub.assign(date="2025-05-29", episode=np.arange(1, len(long_sub)+1))
long_sub[["date","episode","start","end","missing_points","duration_minutes"]].to_csv(TABLES / "longest_blind_window_missing_structure.csv", index=False)

# ---------- zero forecast episodes ----------
zero = df[df["maximum_forecast_demand_mw"].eq(0)].sort_values("timestamp").copy()
zero["newgrp"] = zero["timestamp"].diff().ne(pd.Timedelta(minutes=5)).cumsum()
zrows = []
for gid, sub in zero.groupby("newgrp"):
    zrows.append({
        "episode_id": int(gid), "start": sub["timestamp"].min(), "end": sub["timestamp"].max(),
        "records": len(sub), "duration_minutes": len(sub)*5,
        "minimum_current_demand_mw": sub["current_demand_mw"].min(),
        "maximum_current_demand_mw": sub["current_demand_mw"].max(),
        "minimum_supply_capacity_mw": sub["supply_capacity_mw"].min(),
        "maximum_supply_capacity_mw": sub["supply_capacity_mw"].max(),
    })
zero_ep = pd.DataFrame(zrows)
zero_ep[["episode_id","start","end","records","duration_minutes"]].to_csv(TABLES / "zero_forecast_episodes.csv", index=False)
zero_ep.to_csv(TABLES / "zero_forecast_episodes_detailed.csv", index=False)

# ---------- audit coverage matrix ----------
coverage = pd.DataFrame([
    ["Timestamp sequence","Tested: 917 missing timestamps","Tested: parseable timestamps; 0 duplicates; 0 reverse steps","Not applicable","Not applicable","Tested: 861 missing episodes; longest raw gap 40 min","Not applicable"],
    ["Supply capacity","Tested: 0 null values","Tested: all values positive","Jointly testable through reserve = capacity - demand; six joint failures","Tested with capacity >= demand; all pass","Descriptive screen: max 5-min change 17,460.5 MW; no public hard bound","Descriptive: 955 adjacent repeats; schedule-related repeats not adjudicated"],
    ["Current demand","Tested: 0 null values","Tested: all values positive","Jointly testable through reserve = capacity - demand; six joint failures","Tested with demand <= capacity; all pass","Descriptive screen: archive-max 5-min change 5,197.1 MW at 27 Feb 2025 15:10","Descriptive: 27 adjacent repeats; no failure assigned from repetition alone"],
    ["Maximum forecast demand","Tested: 0 null values","Tested: 125 zero-valued records in 7 episodes","Not testable: no exact public identity with released fields","Not testable: no fixed contemporaneous ordering requirement","Not adjudicated: forecast is stepwise and refresh timing is undocumented","Not adjudicated: repeated daily forecast values are expected by design"],
    ["Supply reserve","Tested: 0 null values","Tested: all values positive","Tested: 6 failures >1 MW; max residual 9,619.68 MW","Tested against operating reserve nesting: 1,316 violations in 105 episodes","Descriptive screen: max 5-min change 17,453.53 MW","Tested contextually: three-value near-freeze occurs inside the six-record identity failure"],
    ["Supply-reserve rate","Tested: 0 null values","Tested: all values positive","Tested against independently calculated reserve rate: 6 failures, co-located with reserve identity failures","Paired rate inequality tested descriptively; not used as an independent proof","Descriptive screen: max 5-min change 22.44 percentage points","No independent freeze classification; rate may inherit source-field defects"],
    ["Operating reserve","Tested: 0 null values","Tested: six zeros; three additional nonzero records below 1,500 MW","Not testable: component reserves are not released","Tested: 1,316 records exceed supply reserve, under the declared reserve-nesting comparison","Tested descriptively: three severe-candidate segments have 10.87-13.24 GW boundary changes","Tested contextually: nearly constant pair, six-record zero run, and isolated low nonzero signature"],
    ["Operating-reserve rate","Tested: 0 null values","Tested: 6 zero values, co-located with operating-reserve zeros","Not independently testable: public metadata do not specify an auditable denominator","Descriptive paired-rate check: 1,173 operating-rate values exceed supply-reserve rate","Descriptive screen: max 5-min change 18.52 percentage points","Co-fails with operating-reserve zero and nearly constant signatures; not independent validation"],
], columns=["Published field or sequence","Existence","Domain","Identity","Inequality","Continuity","Freeze"])
coverage.to_csv(TABLES / "field_rule_coverage_matrix.csv", index=False)


# ---------- additional complete reference outputs ----------
# Compact continuity summary for all contiguous steps.
continuity_full = pd.read_csv(TABLES / "continuity_screen_before_after_exclusions.csv")
continuity_summary = continuity_full[continuity_full["scenario"].eq("All contiguous steps")][[
    "field", "contiguous_steps", "median_abs_change_mw", "p99_abs_change_mw",
    "p9999_abs_change_mw", "maximum_abs_change_mw", "maximum_timestamp", "ratio_to_p9999"
]]
continuity_summary.to_csv(TABLES / "continuity_screen_summary.csv", index=False)

# Top ten absolute changes for each audited MW field in the complete contiguous sample.
top_parts = []
for field in ["supply_capacity_mw", "current_demand_mw", "supply_reserve_mw", "operating_reserve_mw"]:
    sub = df.loc[df["contiguous"], ["timestamp", field, f"{field}_delta", "source_file"]].copy()
    sub = sub.rename(columns={field: "value", f"{field}_delta": "change_mw"})
    sub["abs_change_mw"] = sub["change_mw"].abs()
    sub["field"] = field
    top_parts.append(sub.nlargest(10, "abs_change_mw")[["field", "timestamp", "value", "change_mw", "abs_change_mw", "source_file"]])
pd.concat(top_parts, ignore_index=True).to_csv(TABLES / "continuity_top_changes.csv", index=False)

# Complete reserve-nesting flags and contiguous episodes.
level_violation = df["operating_reserve_mw"].gt(df["supply_reserve_mw"])
level_rows = df[level_violation].copy()
level_rows.to_csv(TABLES / "reserve_nesting_violations.csv", index=False)
level_rows = level_rows.sort_values("timestamp")
level_rows["episode"] = level_rows["timestamp"].diff().ne(pd.Timedelta(minutes=5)).cumsum()
episode_rows = []
for episode_id, sub in level_rows.groupby("episode"):
    episode_rows.append({
        "episode_id": int(episode_id),
        "start": sub["timestamp"].min(),
        "end": sub["timestamp"].max(),
        "records": len(sub),
        "duration_minutes": len(sub) * 5,
        "maximum_excess_mw": sub["reserve_nesting_excess_mw"].max(),
        "minimum_supply_reserve_mw": sub["supply_reserve_mw"].min(),
    })
pd.DataFrame(episode_rows).to_csv(TABLES / "reserve_nesting_violation_episodes.csv", index=False)

# ---------- figures ----------
# Figure 1: audit workflow
fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
ax.set_xlim(0, 15); ax.set_ylim(0, 3); ax.axis("off")
labels = ["Public operational\ndata service", "Coverage\ndeclaration", "Rule-family\nevaluation", "Quality-state\nassignment", "Decision-window\nmonitoring"]
xs = [0.4, 3.3, 6.2, 9.1, 12.0]
for x, label in zip(xs, labels):
    ax.add_patch(Rectangle((x, 1.0), 2.2, 1.0, fill=False, linewidth=1.4))
    ax.text(x+1.1, 1.5, label, ha="center", va="center", fontsize=10)
for a,b in zip(xs[:-1], xs[1:]):
    ax.add_patch(FancyArrowPatch((a+2.2,1.5),(b,1.5),arrowstyle="->",mutation_scale=14,linewidth=1.2))
fig.tight_layout()
fig.savefig(FIGS / "Diagnostic_audit_workflow.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: released operating-reserve patterns
fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.4), dpi=300)
segments_specs = [
    ("2025-02-27 14:40", "2025-02-27 15:35", "(a) Multi-field discontinuity"),
    ("2025-11-17 16:05", "2025-11-17 17:20", "(b) Zero-valued run"),
    ("2026-01-16 14:20", "2026-01-16 15:10", "(c) Isolated low positive value"),
]
for ax, (start, end, title) in zip(axes, segments_specs):
    sub = df[(df.timestamp >= start) & (df.timestamp <= end)]
    ax.plot(sub.timestamp, sub.operating_reserve_mw, "o-", color="#1f4e79", markersize=3, label="Operating reserve")
    ax.plot(sub.timestamp, sub.supply_reserve_mw, "s--", color="#b45309", markersize=3, label="Supply reserve")
    for ts in sub.loc[sub.operating_reserve_mw < 1500, "timestamp"]:
        ax.axvspan(ts - pd.Timedelta(minutes=2.5), ts + pd.Timedelta(minutes=2.5), color="#eeeeee", zorder=0)
    ax.axhline(1500, color="#555555", linestyle=":", linewidth=1, label="1,500 MW screen")
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    ax.set_ylabel("Reserve (MW)")
    clock_axis(ax, start[:10], 10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=9)
fig.tight_layout(h_pad=1.8, rect=(0, 0, 1, .96))
save_publication_figure(fig, "Figure_2_operating_reserve_signatures")
plt.close(fig)

# Figure 3: missingness amplification
avail=pd.DataFrame(availability_rows)
fig,ax=plt.subplots(figsize=(7,4.5),dpi=250)
ax.plot(avail.persistence_minutes,avail.unavailable_share_pct,"o-")
ax.set_xlabel("Required complete window (minutes)")
ax.set_ylabel("Decision-window unavailability (%)")
for x, y in zip(avail.persistence_minutes, avail.unavailable_share_pct):
    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.grid(axis="y",alpha=.25)
fig.tight_layout()
fig.savefig(FIGS / "Diagnostic_missingness_amplification.png",dpi=300,bbox_inches="tight")
plt.close(fig)

# Figure 4: quarterly missingness
fig,ax=plt.subplots(figsize=(8,4.5),dpi=250)
ax.bar(inv.period,inv.missing_per_day)
ax.plot(inv.period,inv.missing_per_day,"o-")
for x, y, n in zip(inv.period, inv.missing_per_day, inv.missing_rows):
    ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
ax.set_ylabel("Missing five-minute records per day")
ax.set_xlabel("Quarter")
ax.tick_params(axis="x",rotation=45)
fig.tight_layout()
fig.savefig(FIGS / "Figure_4_quarterly_missingness.png",dpi=300,bbox_inches="tight")
plt.close(fig)

# Supplementary Figure S1
sub=df[(df.timestamp>="2025-06-17 14:20")&(df.timestamp<="2025-06-17 15:05")]
fig,ax=plt.subplots(figsize=(8,4.5),dpi=250)
ax.plot(sub.timestamp,sub.supply_reserve_mw,"o-",label="Reported supply reserve")
ax.plot(sub.timestamp,sub.calculated_supply_reserve_mw,"s--",label="Capacity minus demand")
ax.axvspan(pd.Timestamp("2025-06-17 14:30"), pd.Timestamp("2025-06-17 14:55"), color="#dceaf2")
ax.set_ylabel("Power (MW)"); ax.legend(frameon=False); clock_axis(ax, "2025-06-17", 5)
fig.tight_layout(); save_publication_figure(fig, "Supplementary_Figure_S1_supply_identity_failure"); plt.close(fig)

# Supplementary Figure S2
level=pd.read_csv(TABLES/"reserve_nesting_by_supply_reserve_level.csv")
time=pd.read_csv(TABLES/"reserve_nesting_by_time_band.csv")
fig,axes=plt.subplots(1,2,figsize=(10,4.5),dpi=250)
level_bars=axes[0].bar(level.supply_reserve_bin_mw,level.level_violations)
axes[0].tick_params(axis="x",rotation=35); axes[0].set_ylabel("Level flags"); axes[0].set_title("(a) By supply-reserve level"); axes[0].set_xlabel("Supply reserve (MW)")
axes[0].set_ylim(0, max(level.level_violations.max()*1.20, 10))
for bar, count, rate in zip(level_bars, level.level_violations, level.level_violation_rate_pct):
    axes[0].annotate(f"{int(count)}\n({rate:.2f}%)", (bar.get_x()+bar.get_width()/2, bar.get_height()), textcoords="offset points", xytext=(0,4), ha="center", fontsize=7)
time_bars=axes[1].bar(time.time_band,time.level_violations)
axes[1].tick_params(axis="x",rotation=35); axes[1].set_title("(b) By time of day"); axes[1].set_xlabel("Time band (KST; UTC+09:00)"); axes[1].set_ylabel("Level flags")
axes[1].set_ylim(0, max(time.level_violations.max()*1.15, 10))
for bar, count in zip(time_bars, time.level_violations):
    axes[1].annotate(str(int(count)), (bar.get_x()+bar.get_width()/2, bar.get_height()), textcoords="offset points", xytext=(0,4), ha="center", fontsize=7)
fig.tight_layout(); save_publication_figure(fig, "Supplementary_Figure_S2_reserve_nesting_level_time"); plt.close(fig)

# Supplementary Figure S3
sub=df[(df.timestamp>="2025-02-27 14:55")&(df.timestamp<="2025-02-27 15:20")]
fig,ax=plt.subplots(figsize=(8,4.5),dpi=250)
for col,label in [("current_demand_mw","Demand"),("supply_reserve_mw","Supply reserve"),("operating_reserve_mw","Operating reserve")]: ax.plot(sub.timestamp,sub[col],"o-",label=label)
ax.axvspan(pd.Timestamp("2025-02-27 15:05"), pd.Timestamp("2025-02-27 15:10"), color="#dceaf2")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82)); ax.set_ylabel("Power (MW)"); clock_axis(ax, "2025-02-27", 5)
fig.tight_layout(); save_publication_figure(fig, "Supplementary_Figure_S3_multifield_snapshot_20250227"); plt.close(fig)

# Supplementary Figure S4
sub=df[(df.timestamp>="2026-06-30 16:20")&(df.timestamp<="2026-06-30 17:10")]
fig,ax=plt.subplots(figsize=(8,4.5),dpi=250)
for col,label in [("supply_capacity_mw","Supply capacity"),("current_demand_mw","Demand"),("supply_reserve_mw","Supply reserve"),("operating_reserve_mw","Operating reserve")]: ax.plot(sub.timestamp,sub[col],"o-",label=label)
ax.legend(frameon=False,ncol=2); ax.set_ylabel("Power (MW)"); clock_axis(ax, "2026-06-30", 10)
fig.tight_layout(); save_publication_figure(fig, "Supplementary_Figure_S4_june30_transition_20260630"); plt.close(fig)

# Key results
key = {
    "observed_rows": int(len(df)),
    "expected_rows": int(len(expected_index)),
    "missing_rows": int(len(missing_index)),
    "missing_share_pct": float(100*len(missing_index)/len(expected_index)),
    "missing_episodes": int(len(missing_episodes)),
    "severe_candidate_records": int((df.operating_reserve_mw<1500).sum()),
    "severe_candidate_segments": int(len(segments)),
    "candidate_severe_minutes": int((df.operating_reserve_mw<1500).sum()*5),
    "supply_identity_failures": int(len(supply_fail)),
    "zero_forecast_records": int(len(zero)),
    "reserve_nesting_level_violations": int((df.operating_reserve_mw>df.supply_reserve_mw).sum()),
    "reserve_nesting_rate_violations": int((df.operating_reserve_rate_pct>df.supply_reserve_rate_pct).sum()),
    "share_level_violations_16_19_59_pct": float(time.loc[time.time_band=="16:00-19:59","share_of_all_level_violations_pct"].iloc[0]),
    "june30_supply_capacity_change_mw": float(df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:35")),"supply_capacity_mw_delta"].iloc[0]),
    "june30_supply_reserve_change_mw": float(df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:35")),"supply_reserve_mw_delta"].iloc[0]),
    "june30_reported_demand_change_mw": float(df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:35")),"current_demand_mw_delta"].iloc[0]),
    "june30_identity_implied_demand_change_mw": float(df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:35")),"supply_capacity_mw_delta"].iloc[0] - df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:35")),"supply_reserve_mw_delta"].iloc[0]),
    "june30_balance_residual_change_mw": float(df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:35")),"supply_identity_residual_mw"].iloc[0] - df.loc[df.timestamp.eq(pd.Timestamp("2026-06-30 16:30")),"supply_identity_residual_mw"].iloc[0]),
    "sixty_minute_unavailable_share_pct": float(avail.loc[avail.persistence_minutes.eq(60),"unavailable_share_pct"].iloc[0]),
    "sixty_minute_longest_blind_run_minutes": int(avail.loc[avail.persistence_minutes.eq(60),"longest_unavailable_run_minutes"].iloc[0]),
}
(OUT/"key_results.json").write_text(json.dumps(key,indent=2),encoding="utf-8")
print(json.dumps(key,indent=2))
