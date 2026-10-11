"""Load the fixed KPX release and reproduce the declared point policies."""
from pathlib import Path
from decimal import Decimal
import os
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path(os.environ.get('KPX_RAW_DIR', str(HERE.parents[1] / 'data_raw')))
_DEFAULT_OUTPUT = HERE.parents[1] / 'reproduced_results' / 'temporal'
OUTPUT_DIR = Path(os.environ.get('KPX_TEMPORAL_OUTPUT', str(_DEFAULT_OUTPUT)))
EXPECTED_RESULTS = DEFAULT_RAW.parent / 'expected_results'
COLUMN_MAP = {
    '공급능력(MW)': 'supply_capacity_mw',
    '현재수요(MW)': 'current_demand_mw',
    '최대예측수요(MW)': 'maximum_forecast_demand_mw',
    '공급예비력(MW)': 'supply_reserve_mw',
    '공급예비율(%)': 'supply_reserve_rate_pct',
    '운영예비력(MW)': 'operating_reserve_mw',
    '운영예비율(%)': 'operating_reserve_rate_pct',
}
FIELDS = list(COLUMN_MAP.values())
IDENTITY_FIELDS = ['supply_capacity_mw', 'current_demand_mw', 'supply_reserve_mw']

def load_release(raw_dir=DEFAULT_RAW):
    parts = []
    paths = sorted(Path(raw_dir).glob('sukub_*.csv'))
    assert len(paths) == 8
    for path in paths:
        part = pd.read_csv(path, encoding='cp949', dtype=str).rename(columns=COLUMN_MAP)
        part['timestamp'] = pd.to_datetime(part['기준일시'], format='%Y%m%d%H%M%S')
        part['source_file'] = path.name
        part['source_row'] = np.arange(1, len(part) + 1)
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    assert len(raw) == 209323 and not raw.timestamp.duplicated().any()
    obs = raw.copy()
    for field in FIELDS:
        obs[field] = obs[field].astype(float)
    # MW strings have no more than four decimal places in these releases.
    # Scaling their exact decimals makes comparisons at every tested threshold exact.
    unit = Decimal('10000')
    ints = {}
    for field in IDENTITY_FIELDS:
        scaled = [Decimal(s) * unit for s in raw[field]]
        assert all(v == v.to_integral_value() for v in scaled)
        ints[field] = np.fromiter((int(v) for v in scaled), dtype=np.int64)
    residual_units = ints['supply_reserve_mw'] - ints['supply_capacity_mw'] + ints['current_demand_mw']
    obs['supply_identity_residual_units'] = residual_units
    obs['supply_identity_residual_mw'] = residual_units / 10000
    index = pd.date_range('2024-07-01', '2026-06-30 23:55:00', freq='5min')
    grid = obs.set_index('timestamp').reindex(index).rename_axis('timestamp')
    grid['observed'] = grid.source_file.notna()
    grid['flag_forecast_zero'] = grid.observed & grid.maximum_forecast_demand_mw.eq(0)
    grid['flag_supply_identity'] = grid.observed & grid.supply_identity_residual_units.abs().gt(10000)
    grid['flag_operating_zero'] = grid.observed & grid.operating_reserve_mw.eq(0)
    grid['flag_low_operating_high_supply'] = grid.observed & grid.operating_reserve_mw.lt(1500) & grid.supply_reserve_mw.ge(5500)
    grid['flag_nesting_ambiguity'] = grid.observed & (grid.operating_reserve_mw - grid.supply_reserve_mw).gt(1e-9)
    grid['flag_unresolved_transition'] = grid.observed & grid.index.to_series().between('2026-06-30 16:20:00', '2026-06-30 17:10:00')
    reserve = grid.observed & ~grid.flag_supply_identity & ~grid.flag_operating_zero & ~grid.flag_low_operating_high_supply & ~grid.flag_unresolved_transition
    missions = {
        'cadence_delivery': grid.observed.to_numpy(),
        'forecast_planning': (grid.observed & ~grid.flag_forecast_zero).to_numpy(),
        'reserve_stage_permissive': reserve.to_numpy(),
        'reserve_stage_strict': (reserve & ~grid.flag_nesting_ambiguity).to_numpy(),
        'full_service_permissive': (reserve & ~grid.flag_forecast_zero).to_numpy(),
        'full_service_strict': (reserve & ~grid.flag_forecast_zero & ~grid.flag_nesting_ambiguity).to_numpy(),
    }
    return raw, obs, grid, missions

def run_arrays(state):
    state = np.asarray(state, dtype=bool)
    starts = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    ends = np.r_[starts[1:] - 1, len(state) - 1]
    return starts, ends, state[starts], ends - starts + 1

def window_state(state, n):
    state = np.asarray(state, dtype=np.int8)
    c = np.r_[0, np.cumsum(1 - state, dtype=np.int64)]
    return (c[n:] - c[:-n]) == 0
