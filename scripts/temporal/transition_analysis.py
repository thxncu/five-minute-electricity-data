"""A temporally calibrated multichannel change screen for public releases.

The screen identifies changes for review. It neither labels physical events nor
asserts that flagged values are wrong. Calibration ends before the test period.
"""
import json
import numpy as np
import pandas as pd
from common import load_release, window_state, run_arrays, HERE, OUTPUT_DIR

FIELDS = ['supply_capacity_mw', 'current_demand_mw', 'operating_reserve_mw']
CALIBRATION_END_EXCLUSIVE = pd.Timestamp('2026-01-01')
PRIMARY_QUANTILE = .999
GAP_MINUTES = 30

def clusters(flags, index, score, train_boundary):
    positions = np.flatnonzero(flags)
    if not len(positions):
        return pd.DataFrame()
    groups = np.r_[0, np.cumsum(np.diff(positions) > GAP_MINUTES//5)]
    rows=[]
    for group in np.unique(groups):
        pos=positions[groups==group]
        first,last=int(pos[0]),int(pos[-1])
        maximum=pos[np.argmax(score[pos])]
        rows.append({'cluster_id': int(group)+1, 'start': index[first], 'end': index[last],
                     'bracket_grid_points': last-first+1, 'flagged_change_points': len(pos),
                     'span_minutes': (last-first)*5, 'peak_timestamp': index[maximum],
                     'maximum_score': score[maximum],
                     'period': 'calibration' if index[last] < train_boundary else 'heldout',
                     'first_grid_position': first, 'last_grid_position': last})
    return pd.DataFrame(rows)

def evaluate_policy(grid, transition_mask, label):
    reserve = grid.observed.to_numpy() & ~grid.flag_supply_identity.to_numpy() & ~grid.flag_operating_zero.to_numpy() & ~grid.flag_low_operating_high_supply.to_numpy() & ~transition_mask
    full = reserve & ~grid.flag_forecast_zero.to_numpy()
    strict = full & ~grid.flag_nesting_ambiguity.to_numpy()
    rows=[]
    for name,z in [('full_service_permissive',full),('full_service_strict',strict)]:
        available=window_state(z,12)
        _,_,v,l=run_arrays(available)
        rows.append({'transition_policy': label, 'mission_code': name,
                     'transition_mask_points':int(transition_mask.sum()),
                     'available_windows':int(available.sum()),
                     'reliability_60min_pct':100*available.mean(),
                     'longest_unavailable_60min_run_minutes': int(l[~v].max())*5})
    return rows

def main():
    out=OUTPUT_DIR;out.mkdir(parents=True,exist_ok=True)
    _,_,grid,_=load_release()
    delta=grid[FIELDS].diff()
    eligible=delta.notna().all(axis=1).to_numpy()
    train=(grid.index<CALIBRATION_END_EXCLUSIVE)&eligible
    test=(grid.index>=CALIBRATION_END_EXCLUSIVE)&eligible
    centers=delta.loc[train].median()
    scales=1.4826*(delta.loc[train]-centers).abs().median()
    assert (scales>0).all()
    standardized=(delta-centers).abs()/scales
    score=np.sort(standardized.to_numpy(),axis=1)[:,1]
    primary_threshold=float(np.quantile(score[train],PRIMARY_QUANTILE,method='linear'))
    flags=eligible&(score>primary_threshold)
    event_table=clusters(flags,grid.index,score,CALIBRATION_END_EXCLUSIVE)
    event_table.to_csv(out/'multichannel_change_clusters.csv',index=False)
    records=pd.concat([grid[FIELDS],delta.add_suffix('_change_mw'),standardized.add_suffix('_standardized_change')],axis=1)
    records['change_score']=score
    records['eligible_adjacent_records']=eligible
    records['primary_threshold']=primary_threshold
    records['flag_change_for_review']=flags
    records.loc[flags].to_csv(out/'multichannel_change_flagged_records.csv')
    records.loc['2026-06-30 14:00':'2026-06-30 19:00'].to_csv(out/'multichannel_change_june30_context.csv')
    contextual=grid.index.to_series().between('2026-06-30 16:20','2026-06-30 17:10').to_numpy()
    nearby=event_table[(event_table.end>=pd.Timestamp('2026-06-30 16:20')) & (event_table.start<=pd.Timestamp('2026-06-30 17:10'))]
    detected_case=np.zeros(len(grid),bool)
    global_brackets=np.zeros(len(grid),bool)
    for _,r in event_table.iterrows():
        global_brackets[int(r.first_grid_position):int(r.last_grid_position)+1]=True
    for _,r in nearby.iterrows():
        detected_case[int(r.first_grid_position):int(r.last_grid_position)+1]=True
    comparisons=[]
    for label,mask in [('no_separate_transition_exclusion',np.zeros(len(grid),bool)),
                       ('contextual_interval_20260630_1620_1710',contextual),
                       ('screen_bracket_for_20260630_case',detected_case),
                       ('all_calibrated_screen_brackets',global_brackets)]:
        comparisons.extend(evaluate_policy(grid,mask,label))
    pd.DataFrame(comparisons).to_csv(out/'multichannel_transition_policy_sensitivity.csv',index=False)
    sensitivity=[]
    for q in [.995,.999,.9995]:
        threshold=float(np.quantile(score[train],q,method='linear'))
        f=eligible&(score>threshold)
        events=clusters(f,grid.index,score,CALIBRATION_END_EXCLUSIVE)
        case=events[(events.end>=pd.Timestamp('2026-06-30 16:20'))&(events.start<=pd.Timestamp('2026-06-30 17:10'))]
        sensitivity.append({'calibration_quantile':q,'score_threshold':threshold,
                            'calibration_change_flags':int(f[train].sum()),'heldout_change_flags':int(f[test].sum()),
                            'all_clusters':len(events),'calibration_clusters':int(events.period.eq('calibration').sum()),
                            'heldout_clusters':int(events.period.eq('heldout').sum()),
                            'june30_cluster_start': str(case.start.min()),'june30_cluster_end':str(case.end.max()),
                            'june30_context_change_flags':int((f&contextual).sum())})
    pd.DataFrame(sensitivity).to_csv(out/'multichannel_change_threshold_sensitivity.csv',index=False)
    summary={'calibration_start':'2024-07-01 00:00:00','calibration_end':'2025-12-31 23:55:00',
             'heldout_start':'2026-01-01 00:00:00','heldout_end':'2026-06-30 23:55:00',
             'fields':FIELDS,'centers_mw':centers.to_dict(),'mad_scales_mw':scales.to_dict(),
             'primary_calibration_quantile':PRIMARY_QUANTILE,'primary_score_threshold':primary_threshold,
             'maximum_between_flag_gap_to_group_minutes':GAP_MINUTES,
             'calibration_eligible_changes':int(train.sum()),'heldout_eligible_changes':int(test.sum()),
             'calibration_flags':int(flags[train].sum()),'heldout_flags':int(flags[test].sum()),
             'calibration_flags_per_1000_eligible':float(1000*flags[train].mean()),
             'heldout_flags_per_1000_eligible':float(1000*flags[test].mean()),
             'all_change_clusters':len(event_table),'heldout_clusters':int(event_table.period.eq('heldout').sum()),
             'contextual_interval_points':int(contextual.sum()),'detected_case_bracket_points':int(detected_case.sum()),
             'detected_case_intersection_with_contextual_points':int((detected_case&contextual).sum()),
             'contextual_points_outside_detected_bracket':int((contextual&~detected_case).sum()),
             'detected_case_start': str(grid.index[detected_case].min()),'detected_case_end':str(grid.index[detected_case].max()),
             'june30_flag_timestamps':[str(t) for t in grid.index[flags&contextual]],
             'threshold_sensitivity':sensitivity,
             'limitations':'This review screen detects coincident changes, not a persistent level change or a physical/data fault. The 30-minute grouping convention brackets flagged changes and does not estimate event recovery. No labelled normal-event set is available, so the flag rate is review burden, not false-alarm rate. Calibration is temporally separate but retrospective; no prospective performance is claimed.'}
    (out/'transition_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':
    main()
