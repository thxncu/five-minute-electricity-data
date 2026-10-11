"""Exact decimal tolerance sensitivity and observed publication precision."""
from pathlib import Path
from decimal import Decimal
import json
import numpy as np
import pandas as pd
from common import load_release, IDENTITY_FIELDS, OUTPUT_DIR

def main():
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    raw, obs, _, _ = load_release()
    r = obs.supply_identity_residual_units.to_numpy(dtype=np.int64)
    absolute = np.abs(r)
    tolerances = ['0.1', '0.2', '0.5', '0.51', '0.6', '1', '5', '10', '100', '1000']
    rows = []
    for tau in tolerances:
        t = int(Decimal(tau) * 10000)
        over = absolute > t
        rows.append({
            'tolerance_mw': tau,
            'records_exceeding_tolerance': int(over.sum()),
            'minimum_residual_above_tolerance_mw': float(absolute[over].min() / 10000) if over.any() else None,
            'maximum_residual_at_or_below_tolerance_mw': float(absolute[~over].max() / 10000) if (~over).any() else None,
            'records_exactly_on_tolerance': int((absolute == t).sum()),
        })
    pd.DataFrame(rows).to_csv(out / 'supply_identity_tolerance_exact_decimal.csv', index=False)
    value_counts = pd.Series(r).value_counts().sort_index()
    pd.DataFrame({'residual_mw': value_counts.index / 10000, 'records': value_counts.values}).to_csv(out / 'supply_identity_residual_exact_distribution.csv', index=False)
    precision = []
    six_sig_mask = np.ones(len(raw), dtype=bool)
    field_steps = {}
    for field in IDENTITY_FIELDS:
        vals = [Decimal(s) for s in raw[field]]
        significant_digits = np.array([len(v.normalize().as_tuple().digits) for v in vals])
        places = np.array([max(0, -v.as_tuple().exponent) for v in vals])
        six_sig_mask &= significant_digits <= 6
        # A six-significant-digit rounded value has unit 10^(order-5).
        step = np.array([10.0 ** (v.adjusted() - 5) for v in vals])
        field_steps[field] = step
        for sig in np.unique(significant_digits):
            precision.append({'field': field, 'measure': 'significant_digits_required', 'value': int(sig), 'records': int((significant_digits == sig).sum())})
        for dec in np.unique(places):
            precision.append({'field': field, 'measure': 'literal_decimal_places', 'value': int(dec), 'records': int((places == dec).sum())})
    pd.DataFrame(precision).to_csv(out / 'publication_precision_distribution.csv', index=False)
    envelope = .5 * sum(field_steps.values())
    rounding_compatible = np.abs(r / 10000) <= envelope + 1e-10
    diagnostics = obs[['timestamp', 'source_file', 'source_row', *IDENTITY_FIELDS]].copy()
    diagnostics['residual_mw'] = r / 10000
    diagnostics['all_three_fields_at_most_six_significant_digits'] = six_sig_mask
    diagnostics['six_significant_digit_rounding_envelope_mw'] = envelope
    diagnostics['within_rounding_envelope'] = rounding_compatible
    diagnostics.to_csv(out / 'publication_precision_record_diagnostics.csv', index=False)
    report = {
        'observed_records': len(obs), 'decimal_integer_scale_per_mw': 10000,
        'six_significant_digits_all_three_fields_records': int(six_sig_mask.sum()),
        'six_significant_digits_all_three_fields_pct': float(100 * six_sig_mask.mean()),
        'more_precision_records': int((~six_sig_mask).sum()),
        'within_six_significant_rounding_envelope_all_records': int(rounding_compatible.sum()),
        'outside_six_significant_rounding_envelope_all_records': int((~rounding_compatible).sum()),
        'maximum_absolute_residual_not_above_one_mw': float((absolute[absolute <= 10000]).max() / 10000),
        'minimum_absolute_residual_above_one_mw': float((absolute[absolute > 10000]).min() / 10000),
        'exactly_half_mw_records': int((absolute == 5000).sum()),
        'tolerance_sensitivity': rows,
        'interpretation': 'Observed formatting is consistent with six-significant-digit rounding for almost all records. A rounding-to-nearest convention is not independently documented. One MW is a conservative publication-precision tolerance, not a physical deadband or an optimized threshold.'
    }
    (out / 'rounding_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
