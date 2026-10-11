"""Exact finite-run identities and an observed-state Markov reference.

The stochastic calculations do not replace observed-window counts. Parameters
estimated before 2026 are also evaluated on the subsequent six months.
"""
import json
import numpy as np
import pandas as pd
from common import load_release, run_arrays, window_state, HERE, OUTPUT_DIR

SEED = 20261011
WINDOW_SLOTS = np.array([1, 2, 3, 6, 12, 24, 48])
N_RESAMPLES = 1000
BLOCK_DAYS = 7

def fit_markov(z):
    z = np.asarray(z, bool)
    good_prev = z[:-1]
    bad_prev = ~good_prev
    gg = int((good_prev & z[1:]).sum())
    gb = int((good_prev & ~z[1:]).sum())
    bg = int((bad_prev & z[1:]).sum())
    bb = int((bad_prev & ~z[1:]).sum())
    a = gb / (gg + gb)
    b = bg / (bg + bb)
    pi_good = b / (a + b)
    return {'n': len(z), 'gg': gg, 'gb': gb, 'bg': bg, 'bb': bb,
            'a_good_to_bad': a, 'b_bad_to_good': b, 'pi_good': pi_good,
            'pi_bad': 1-pi_good, 'observed_point_bad_share': float((~z).mean()),
            'lag_one_correlation_markov': 1-a-b,
            'mean_bad_run_slots_markov': 1/b, 'mean_good_run_slots_markov': 1/a}

def markov_unavailable(fit, n):
    return 1 - fit['pi_good'] * (1 - fit['a_good_to_bad']) ** (np.asarray(n) - 1)

def finite_run_statistics(z, n):
    starts, ends, values, lengths = run_arrays(z)
    total_n = len(z)
    m = total_n - n + 1
    available = int(np.maximum(lengths[values] - n + 1, 0).sum())
    actual = window_state(z, n)
    assert int(actual.sum()) == available
    bad_count = int((~z).sum())
    bad_runs = int((~values).sum())
    lower = min(m, bad_count)
    upper = min(m, bad_count + (n-1)*bad_runs)
    assert lower <= m-available <= upper
    _, _, win_values, win_lengths = run_arrays(actual)
    longest = int(win_lengths[~win_values].max()) if (~win_values).any() else 0
    return {'total_points': total_n, 'window_slots': n, 'eligible_windows': m,
            'available_windows_exact_good_runs': available,
            'unavailable_windows': m-available, 'unavailability_pct': 100*(m-available)/m,
            'bad_points': bad_count, 'bad_runs': bad_runs,
            'finite_lower_bound_pct': 100*lower/m, 'finite_upper_bound_pct': 100*upper/m,
            'longest_unavailable_run_minutes': longest*5}

def run_model_summary(z, fit, mission):
    starts, ends, values, lengths = run_arrays(z)
    rows = []
    for value, exit_p, label in [(True, fit['a_good_to_bad'], 'good'), (False, fit['b_bad_to_good'], 'bad')]:
        complete = (values == value) & (starts > 0) & (ends < len(z)-1)
        lens = lengths[complete]
        for t in [1, 2, 3, 6, 12, 24, 48, 144, 288, 576, 2016]:
            rows.append({'mission_code': mission, 'run_state': label, 'minimum_run_slots': t,
                         'complete_runs': len(lens), 'runs_at_least_threshold': int((lens >= t).sum()),
                         'observed_survival': float((lens >= t).mean()),
                         'geometric_survival': float((1-exit_p)**(t-1)),
                         'empirical_mean_slots': float(lens.mean()),
                         'empirical_cv': float(lens.std(ddof=1)/lens.mean()),
                         'geometric_cv': float(np.sqrt(1-exit_p)),
                         'maximum_complete_run_slots': int(lens.max())})
    return rows

def longest_from_runs(starts, ends, values, n, total_n):
    bad = ~values
    left = np.maximum(n-1, starts[bad])
    right = np.minimum(total_n-1, ends[bad]+n-1)
    if not len(left):
        return 0
    current_left = int(left[0]); current_right = int(right[0]); best = 0
    for s, e in zip(left[1:], right[1:]):
        if s <= current_right+1:
            current_right = max(current_right, int(e))
        else:
            best = max(best, current_right-current_left+1)
            current_left, current_right = int(s), int(e)
    return max(best, current_right-current_left+1)

def simulate_markov_summary(fit, total_n, gen, n=12):
    a, b = fit['a_good_to_bad'], fit['b_bad_to_good']
    first_good = gen.random() < fit['pi_good']
    r_size = max(200, int(4 * total_n * a * b / (a+b) + 200))
    while True:
        values = (np.arange(r_size) % 2 == 0) == first_good
        lengths = gen.geometric(np.where(values, a, b))
        ends_plus = np.cumsum(lengths)
        if ends_plus[-1] >= total_n:
            break
        r_size *= 2
    k = int(np.searchsorted(ends_plus, total_n))+1
    values, lengths, ends_plus = values[:k], lengths[:k], ends_plus[:k]
    lengths[-1] -= int(ends_plus[-1]-total_n)
    ends = np.cumsum(lengths)-1
    starts = np.r_[0, ends[:-1]+1]
    good_count = int(np.maximum(lengths[values]-n+1, 0).sum())
    return [100*(1-good_count/(total_n-n+1)),
            5*longest_from_runs(starts, ends, values, n, total_n),
            int(lengths[~values].max()), int(lengths[values].max())]

def daily_bootstrap_reference(z, index, fit, gen):
    """Resample seven-day blocks of observed daily sufficient statistics.

    Transitions are assigned by their ending day, so resampling does not create
    artificial transitions at joins. Intervals describe pooled-period sampling
    variability conditional on this dependence approximation.
    """
    table = pd.DataFrame({'timestamp': index[1:],
                          'gg': z[:-1] & z[1:], 'gb': z[:-1] & ~z[1:],
                          'bg': ~z[:-1] & z[1:], 'bb': ~z[:-1] & ~z[1:]})
    daily = table.groupby(table.timestamp.dt.normalize())[['gg', 'gb', 'bg', 'bb']].sum().to_numpy()
    d = len(daily)
    samples = []
    for _ in range(N_RESAMPLES):
        starts = gen.integers(0, d-BLOCK_DAYS+1, size=int(np.ceil(d/BLOCK_DAYS)))
        sampled = (starts[:, None] + np.arange(BLOCK_DAYS)).ravel()[:d]
        gg, gb, bg, bb = daily[sampled].sum(axis=0)
        a = gb/(gg+gb); b = bg/(bg+bb); pi_good = b/(a+b)
        samples.append([a, b, 1-pi_good, 100*(1-pi_good*(1-a)**11)])
    samples = np.asarray(samples)
    labels = ['a_good_to_bad', 'b_bad_to_good', 'pi_bad', 'markov_60min_unavailability_pct']
    return {label: {'low_2_5': float(np.quantile(samples[:,j], .025)), 'median': float(np.median(samples[:,j])), 'high_97_5': float(np.quantile(samples[:,j], .975))} for j, label in enumerate(labels)}

def main():
    out = OUTPUT_DIR; out.mkdir(parents=True, exist_ok=True)
    _, _, grid, missions = load_release()
    fit_rows, window_rows, run_rows, quarter_rows, test_rows = [], [], [], [], []
    uncertainty, simulation_summary = {}, {}
    gen = np.random.default_rng(SEED)
    train_mask = grid.index < pd.Timestamp('2026-01-01')
    split = int(train_mask.sum())
    for mission, z in missions.items():
        for period, zz in [('pooled_24_months', z), ('train_18_months', z[:split]), ('test_6_months', z[split:])]:
            fit = fit_markov(zz)
            fit_rows.append({'mission_code': mission, 'period': period, **fit})
            for n in WINDOW_SLOTS:
                exact = finite_run_statistics(zz, int(n))
                q = fit['observed_point_bad_share']
                row = {'mission_code': mission, 'period': period, 'window_minutes': int(n*5), **exact,
                       'markov_unavailability_pct': float(100*markov_unavailable(fit, n)),
                       'iid_unavailability_pct': float(100*(1-(1-q)**n))}
                row['markov_minus_observed_pp'] = row['markov_unavailability_pct'] - row['unavailability_pct']
                row['iid_minus_observed_pp'] = row['iid_unavailability_pct'] - row['unavailability_pct']
                window_rows.append(row)
        pooled_fit = fit_markov(z)
        run_rows.extend(run_model_summary(z, pooled_fit, mission))
        fitted = fit_markov(z[:split])
        q_train = fitted['observed_point_bad_share']
        for n in WINDOW_SLOTS:
            entire_windows = window_state(z, int(n))
            test_endings = grid.index[n-1:] >= pd.Timestamp('2026-01-01')
            actual = float((~entire_windows[test_endings]).mean())
            test_rows.append({'mission_code': mission, 'window_minutes': int(n*5),
                              'test_eligible_endings_with_prehistory': int(test_endings.sum()),
                              'test_actual_unavailability_pct': 100*actual,
                              'train_fitted_markov_prediction_pct': float(100*markov_unavailable(fitted, n)),
                              'train_fitted_iid_prediction_pct': float(100*(1-(1-q_train)**n)),
                              'markov_prediction_minus_observed_pp': float(100*(markov_unavailable(fitted, n)-actual))})
        for q_period in grid.index.to_period('Q').unique():
            mask = grid.index.to_period('Q') == q_period
            f = fit_markov(z[mask])
            quarter_rows.append({'mission_code': mission, 'quarter': str(q_period), **f})
        uncertainty[mission] = {
            'pooled_7day_block_resampling': daily_bootstrap_reference(z, grid.index, pooled_fit, gen),
            'training_7day_block_resampling': daily_bootstrap_reference(z[:split], grid.index[:split], fitted, gen),
        }
        simulations = np.array([simulate_markov_summary(pooled_fit, len(z), gen) for _ in range(N_RESAMPLES)])
        pd.DataFrame(simulations, columns=['unavailability_60min_pct', 'longest_unavailable_60min_minutes', 'longest_bad_point_run_slots', 'longest_good_point_run_slots']).to_csv(out / f'markov_simulations_{mission}.csv', index=False)
        simulation_summary[mission] = {name: {'low_2_5': float(np.quantile(simulations[:,i], .025)), 'median': float(np.median(simulations[:,i])), 'high_97_5': float(np.quantile(simulations[:,i], .975))} for i, name in enumerate(['unavailability_60min_pct', 'longest_unavailable_60min_minutes', 'longest_bad_point_run_slots', 'longest_good_point_run_slots'])}
    pd.DataFrame(fit_rows).to_csv(out / 'markov_transition_parameters.csv', index=False)
    pd.DataFrame(window_rows).to_csv(out / 'markov_and_exact_window_comparison.csv', index=False)
    pd.DataFrame(run_rows).to_csv(out / 'markov_run_length_diagnostics.csv', index=False)
    pd.DataFrame(quarter_rows).to_csv(out / 'markov_quarterly_parameters.csv', index=False)
    pd.DataFrame(test_rows).to_csv(out / 'markov_chronological_holdout.csv', index=False)
    summary = {'seed': SEED, 'n_resamples': N_RESAMPLES, 'bootstrap_block_days': BLOCK_DAYS,
               'train_start': str(grid.index[0]), 'train_end': str(grid.index[split-1]),
               'test_start': str(grid.index[split]), 'test_end': str(grid.index[-1]),
               'resampling_method': 'Moving blocks of seven consecutive calendar days, without circular wrap-around; daily transition sufficient statistics retain within-day transitions and original boundary pairs.',
               'uncertainty_reference_intervals': uncertainty, 'parametric_simulation_reference': simulation_summary,
               'caution': 'A pooled stationary first-order Markov chain is an interpretable reference, not an established mechanism or future reliability guarantee. Overlapping windows are not treated as independent. Seven-day blocks retain short-range dependence but do not remove seasonal changes or guarantee interval coverage under nonstationarity.'}
    (out / 'markov_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    select = pd.DataFrame(window_rows)
    print(select[(select.period=='pooled_24_months') & (select.window_minutes==60)].to_string(index=False))
    print(pd.DataFrame(test_rows).query('window_minutes == 60').to_string(index=False))
    print(json.dumps({k:v for k,v in simulation_summary.items() if k == 'cadence_delivery'}, indent=2))

if __name__ == '__main__':
    main()
