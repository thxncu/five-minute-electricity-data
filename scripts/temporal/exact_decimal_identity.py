"""Exact public-decimal supply identity for the KPX release schema.

Use the source strings when available. Decimal(str(value)) also exactly recovers
the at-most-nine-significant-digit numeric values in the eight fixed releases.
The scale records the finest observed precision: 0.0001 MW.
"""
from decimal import Decimal
import numpy as np

UNITS_PER_MW = 10000

def identity_residual_units(capacity, demand, reserve):
    scale = Decimal(UNITS_PER_MW)
    rows = []
    for c, d, r in zip(capacity, demand, reserve):
        v = (Decimal(str(r)) - Decimal(str(c)) + Decimal(str(d))) * scale
        if v != v.to_integral_value():
            raise ValueError('Publication precision exceeds the declared 0.0001 MW integer scale')
        rows.append(int(v))
    return np.asarray(rows, dtype=np.int64)

def tolerance_mask(residual_units, tolerance_mw):
    cutoff = Decimal(str(tolerance_mw)) * Decimal(UNITS_PER_MW)
    if cutoff != cutoff.to_integral_value():
        raise ValueError('Tolerance does not match the declared integer MW scale')
    return np.abs(np.asarray(residual_units, dtype=np.int64)) > int(cutoff)
