from __future__ import annotations

from pathlib import Path
from decimal import Decimal
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
DERIVED = ROOT / "derived_data"
DERIVED.mkdir(exist_ok=True)

COLUMN_MAP = {
    "공급능력(MW)": "supply_capacity_mw",
    "현재수요(MW)": "current_demand_mw",
    "최대예측수요(MW)": "maximum_forecast_demand_mw",
    "공급예비력(MW)": "supply_reserve_mw",
    "공급예비율(%)": "supply_reserve_rate_pct",
    "운영예비력(MW)": "operating_reserve_mw",
    "운영예비율(%)": "operating_reserve_rate_pct",
}

files = sorted(RAW.glob("sukub_*.csv"))
if len(files) != 8:
    raise RuntimeError(f"Expected eight quarterly source files, found {len(files)}")

parts = []
for file_index, path in enumerate(files):
    frame = pd.read_csv(path, encoding="cp949", dtype=str)
    required = ["기준일시", *COLUMN_MAP.keys()]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame["source_file"] = path.name
    frame["source_row"] = np.arange(1, len(frame) + 1)
    frame["file_index"] = file_index
    frame = frame.rename(columns=COLUMN_MAP)
    # Evaluate the published arithmetic relation in its original decimal representation.
    frame["supply_identity_residual_mw"] = [
        float(Decimal(r) - Decimal(c) + Decimal(d))
        for r, c, d in zip(frame["supply_reserve_mw"], frame["supply_capacity_mw"], frame["current_demand_mw"])
    ]
    for field in COLUMN_MAP.values():
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    frame["timestamp"] = pd.to_datetime(
        frame["기준일시"].astype(str), format="%Y%m%d%H%M%S", errors="raise"
    )
    parts.append(frame)

panel = (
    pd.concat(parts, ignore_index=True)
    .sort_values("timestamp", kind="stable")
    .reset_index(drop=True)
)

identity_residual = panel.pop("supply_identity_residual_mw")

fields = [
    "supply_capacity_mw",
    "current_demand_mw",
    "maximum_forecast_demand_mw",
    "supply_reserve_mw",
    "supply_reserve_rate_pct",
    "operating_reserve_mw",
    "operating_reserve_rate_pct",
]
for field in fields:
    panel[f"{field}_delta"] = panel[field].diff()

panel["dt_min"] = panel["timestamp"].diff().dt.total_seconds() / 60
panel["contiguous"] = panel["dt_min"].eq(5)
panel["calculated_supply_reserve_mw"] = (
    panel["supply_capacity_mw"] - panel["current_demand_mw"]
)
panel["supply_identity_residual_mw"] = identity_residual
panel["calculated_supply_reserve_rate_pct"] = (
    100 * panel["calculated_supply_reserve_mw"] / panel["current_demand_mw"]
)
panel["supply_rate_independent_residual_pp"] = (
    panel["supply_reserve_rate_pct"] - panel["calculated_supply_reserve_rate_pct"]
)
panel["reserve_nesting_excess_mw"] = (
    panel["operating_reserve_mw"] - panel["supply_reserve_mw"]
)

out = DERIVED / "audited_panel.csv"
panel.to_csv(out, index=False)
print(f"Wrote {out.relative_to(ROOT)} with {len(panel):,} rows")
