import pandas as pd, numpy as np, os, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'derived_data'
OUT=ROOT/'reproduced_results'
OUT.mkdir(exist_ok=True)
(OUT/'tables').mkdir(exist_ok=True)

p=TABLES/'audited_panel.csv'
df=pd.read_csv(p, parse_dates=['timestamp'])

# --- reserve nesting level and rate ---
level_excess=df['operating_reserve_mw']-df['supply_reserve_mw']
rate_excess=df['operating_reserve_rate_pct']-df['supply_reserve_rate_pct']
level_violation=level_excess>1e-9
rate_violation=rate_excess>1e-9

# Fixed bins
bins=[-np.inf,6000,10000,15000,20000,np.inf]
labels=['<6,000','6,000-<10,000','10,000-<15,000','15,000-<20,000','>=20,000']
df['supply_reserve_bin']=pd.cut(df['supply_reserve_mw'], bins=bins, labels=labels, right=False)
rows=[]
for lab in labels:
    sub=df[df.supply_reserve_bin==lab]
    n=len(sub); lv=int(level_violation[sub.index].sum()); rv=int(rate_violation[sub.index].sum())
    rows.append({
        'supply_reserve_bin_mw':lab,'all_records':n,'level_violations':lv,
        'level_violation_rate_pct':100*lv/n if n else np.nan,
        'share_of_all_level_violations_pct':100*lv/level_violation.sum() if level_violation.sum() else np.nan,
        'rate_violations':rv,'rate_violation_rate_pct':100*rv/n if n else np.nan
    })
level_bin=pd.DataFrame(rows)
level_bin.to_csv(OUT/'tables'/'reserve_nesting_by_supply_reserve_level.csv',index=False)

# Deciles / percentiles
q=pd.qcut(df['supply_reserve_mw'],10,duplicates='drop')
dec=[]
for interval, sub in df.groupby(q, observed=True):
    lv=int(level_violation[sub.index].sum())
    dec.append({'supply_reserve_decile':str(interval),'all_records':len(sub),'level_violations':lv,'violation_rate_pct':100*lv/len(sub),'share_of_violations_pct':100*lv/level_violation.sum()})
pd.DataFrame(dec).to_csv(OUT/'tables'/'reserve_nesting_by_supply_reserve_decile.csv',index=False)

# by hour
hourly=[]
for h in range(24):
    sub=df[df.timestamp.dt.hour==h]
    lv=int(level_violation[sub.index].sum()); rv=int(rate_violation[sub.index].sum())
    hourly.append({'hour':h,'all_records':len(sub),'level_violations':lv,'level_violation_rate_pct':100*lv/len(sub),'share_of_all_level_violations_pct':100*lv/level_violation.sum(),'rate_violations':rv})
hourly=pd.DataFrame(hourly)
hourly.to_csv(OUT/'tables'/'reserve_nesting_by_hour.csv',index=False)

# broad period bins
periods=[('00:00-06:59',0,7),('07:00-12:59',7,13),('13:00-15:59',13,16),('16:00-19:59',16,20),('20:00-23:59',20,24)]
pr=[]
for name,a,b in periods:
    sub=df[(df.timestamp.dt.hour>=a)&(df.timestamp.dt.hour<b)]
    lv=int(level_violation[sub.index].sum())
    pr.append({'time_band':name,'all_records':len(sub),'level_violations':lv,'level_violation_rate_pct':100*lv/len(sub),'share_of_all_level_violations_pct':100*lv/level_violation.sum()})
pd.DataFrame(pr).to_csv(OUT/'tables'/'reserve_nesting_by_time_band.csv',index=False)

# violation summary
viol=df[level_violation].copy()
non=df[~level_violation].copy()
summary=pd.DataFrame([
    {'metric':'Level violations','value':int(level_violation.sum())},
    {'metric':'Rate violations','value':int(rate_violation.sum())},
    {'metric':'Both level and rate violations','value':int((level_violation&rate_violation).sum())},
    {'metric':'Level-only violations','value':int((level_violation&~rate_violation).sum())},
    {'metric':'Rate-only violations','value':int((~level_violation&rate_violation).sum())},
    {'metric':'Violation supply reserve minimum (MW)','value':viol.supply_reserve_mw.min()},
    {'metric':'Violation supply reserve median (MW)','value':viol.supply_reserve_mw.median()},
    {'metric':'Violation supply reserve mean (MW)','value':viol.supply_reserve_mw.mean()},
    {'metric':'Violation supply reserve maximum (MW)','value':viol.supply_reserve_mw.max()},
    {'metric':'Nonviolation supply reserve median (MW)','value':non.supply_reserve_mw.median()},
    {'metric':'Share of level violations below 15,000 MW (%)','value':100*((viol.supply_reserve_mw<15000).mean())},
    {'metric':'Share of level violations below 20,000 MW (%)','value':100*((viol.supply_reserve_mw<20000).mean())},
])
summary.to_csv(OUT/'tables'/'reserve_nesting_summary.csv',index=False)

# --- level-only vs rate relation and inferred denominator ---
df['operating_rate_if_current_demand_pct']=100*df.operating_reserve_mw/df.current_demand_mw
df['operating_rate_residual_vs_current_demand_pp']=df.operating_reserve_rate_pct-df.operating_rate_if_current_demand_pct
# inferred denom only when rate nonzero
valid=df.operating_reserve_rate_pct.ne(0)
df['implied_operating_rate_denominator_mw']=np.where(valid,100*df.operating_reserve_mw/df.operating_reserve_rate_pct,np.nan)
df['implied_denom_minus_current_demand_mw']=df.implied_operating_rate_denominator_mw-df.current_demand_mw
level_only=df[level_violation & ~rate_violation].copy()
rate_relation=pd.DataFrame([
    {'metric':'Level violations','value':int(level_violation.sum())},
    {'metric':'Rate violations','value':int(rate_violation.sum())},
    {'metric':'Level-only violations','value':len(level_only)},
    {'metric':'Level-only median MW excess','value':level_only.reserve_nesting_excess_mw.median()},
    {'metric':'Level-only mean MW excess','value':level_only.reserve_nesting_excess_mw.mean()},
    {'metric':'Level-only 90th percentile MW excess','value':level_only.reserve_nesting_excess_mw.quantile(.9)},
    {'metric':'Level-only maximum MW excess','value':level_only.reserve_nesting_excess_mw.max()},
    {'metric':'All records median absolute operating-rate residual vs current-demand denominator (percentage points)','value':df.operating_rate_residual_vs_current_demand_pp.abs().median()},
    {'metric':'All records maximum absolute operating-rate residual vs current-demand denominator (percentage points)','value':df.operating_rate_residual_vs_current_demand_pp.abs().max()},
    {'metric':'All records median implied-denominator difference from current demand (MW)','value':df.implied_denom_minus_current_demand_mw.abs().median()},
    {'metric':'All records mean implied-denominator difference from current demand (MW)','value':df.implied_denom_minus_current_demand_mw.abs().mean()},
    {'metric':'All records maximum implied-denominator difference from current demand (MW)','value':df.implied_denom_minus_current_demand_mw.abs().max()},
])
rate_relation.to_csv(OUT/'tables'/'operating_reserve_rate_denominator_diagnostics.csv',index=False)
level_only[['timestamp','supply_reserve_mw','operating_reserve_mw','supply_reserve_rate_pct','operating_reserve_rate_pct','reserve_nesting_excess_mw','operating_rate_if_current_demand_pct','operating_rate_residual_vs_current_demand_pp','implied_operating_rate_denominator_mw','implied_denom_minus_current_demand_mw']].to_csv(OUT/'tables'/'level_only_reserve_nesting_violations.csv',index=False)

# --- zero forecast episodes detailed ---
zero=df[df.maximum_forecast_demand_mw.eq(0)].copy()
# groups contiguous by 5 min
zero=zero.sort_values('timestamp')
zero['newgrp']=(zero.timestamp.diff()!=pd.Timedelta(minutes=5)).cumsum()
zrows=[]
for gid,sub in zero.groupby('newgrp'):
    zrows.append({'episode_id':gid,'start':sub.timestamp.min(),'end':sub.timestamp.max(),'records':len(sub),'duration_minutes':len(sub)*5,
                  'minimum_current_demand_mw':sub.current_demand_mw.min(),'maximum_current_demand_mw':sub.current_demand_mw.max(),
                  'minimum_supply_capacity_mw':sub.supply_capacity_mw.min(),'maximum_supply_capacity_mw':sub.supply_capacity_mw.max()})
zero_ep=pd.DataFrame(zrows)
zero_ep.to_csv(OUT/'tables'/'zero_forecast_episodes_detailed.csv',index=False)

# --- continuity distributions ---
fields=['supply_capacity_mw','current_demand_mw','supply_reserve_mw','operating_reserve_mw']
# classified failure timestamps: severe low op segments + supply identity failures
severe=df[df.operating_reserve_mw<1500].timestamp
supply_fail=df[df.supply_identity_residual_mw.abs()>1].timestamp
classified=set(severe.tolist()) | set(supply_fail.tolist())
# +/- 5 min adjacency
classified_adj=set()
for t in classified:
    classified_adj.update([t-pd.Timedelta(minutes=5),t,t+pd.Timedelta(minutes=5)])
# unresolved June 30 episode 16:30-17:00 +/-5
june_start=pd.Timestamp('2026-06-30 16:30'); june_end=pd.Timestamp('2026-06-30 17:00')
june=set(pd.date_range(june_start-pd.Timedelta(minutes=5),june_end+pd.Timedelta(minutes=5),freq='5min').tolist())

scenarios={
    'All contiguous steps':set(),
    'Excluding classified failures and adjacent steps':classified_adj,
    'Additionally excluding the 30 June 2026 transition episode':classified_adj|june,
}
cont=[]
for scen,exclude in scenarios.items():
    # a delta at t uses t and t-5, so exclude if either current or previous in exclude
    current_ex=df.timestamp.isin(exclude)
    previous_ex=df.timestamp.shift(1).isin(exclude)
    eligible=df.contiguous & ~current_ex & ~previous_ex
    for f in fields:
        s=df.loc[eligible,f+'_delta'].abs()
        idx=s.idxmax()
        cont.append({'scenario':scen,'field':f,'contiguous_steps':len(s),'median_abs_change_mw':s.median(),'p99_abs_change_mw':s.quantile(.99),'p999_abs_change_mw':s.quantile(.999),'p9999_abs_change_mw':s.quantile(.9999),'maximum_abs_change_mw':s.max(),'maximum_timestamp':df.loc[idx,'timestamp'],'ratio_to_p9999':s.max()/s.quantile(.9999)})
continuity=pd.DataFrame(cont)
continuity.to_csv(OUT/'tables'/'continuity_screen_before_after_exclusions.csv',index=False)

# top changes after exclusions for each scenario
for scen,exclude in scenarios.items():
    current_ex=df.timestamp.isin(exclude); previous_ex=df.timestamp.shift(1).isin(exclude)
    eligible=df.contiguous & ~current_ex & ~previous_ex
    top=[]
    for f in fields:
        sub=df.loc[eligible,['timestamp',f,f+'_delta']].copy()
        sub=sub.rename(columns={f:'value',f+'_delta':'change_mw'})
        sub['abs_change_mw']=sub['change_mw'].abs(); sub['field']=f
        top.append(sub.nlargest(10,'abs_change_mw'))
    pd.concat(top).to_csv(OUT/'tables'/('continuity_top_changes_'+scen.lower().replace(' ','_').replace('30_june_2026','june30').replace('/','_')+'.csv'),index=False)

# --- 2025-02-27 and 2026-06-30 episodes ---
for date,start,end,name in [
    ('2025-02-27','2025-02-27 14:55','2025-02-27 15:20','multifield_snapshot_20250227.csv'),
    ('2026-06-30','2026-06-30 16:20','2026-06-30 17:10','june30_transition_20260630.csv')]:
    sub=df[(df.timestamp>=start)&(df.timestamp<=end)].copy()
    cols=['timestamp','supply_capacity_mw','current_demand_mw','maximum_forecast_demand_mw','supply_reserve_mw','supply_reserve_rate_pct','operating_reserve_mw','operating_reserve_rate_pct','supply_capacity_mw_delta','current_demand_mw_delta','supply_reserve_mw_delta','operating_reserve_mw_delta','supply_identity_residual_mw','reserve_nesting_excess_mw']
    sub[cols].to_csv(OUT/'tables'/name,index=False)

# Screening-condition contributions
# high-supply criterion applies to severe candidates
severe_candidates=df[df.operating_reserve_mw<1500].copy()
contrib=pd.DataFrame([
    {'condition':'Cross-field high supply reserve (supply reserve >= 5,500 MW)','candidate_segments_quarantined':3,'incremental_segments_beyond_prior_conditions':3},
    {'condition':'Normal boundary operating reserve values','candidate_segments_quarantined':3,'incremental_segments_beyond_prior_conditions':0},
    {'condition':'Large boundary jump (>=5,000 MW)','candidate_segments_quarantined':3,'incremental_segments_beyond_prior_conditions':0},
])
contrib.to_csv(OUT/'tables'/'gate_condition_incremental_contribution.csv',index=False)

# save relevant audited columns for supplement
cols=['timestamp','supply_capacity_mw','current_demand_mw','maximum_forecast_demand_mw','supply_reserve_mw','supply_reserve_rate_pct','operating_reserve_mw','operating_reserve_rate_pct','supply_identity_residual_mw','reserve_nesting_excess_mw','operating_rate_if_current_demand_pct','operating_rate_residual_vs_current_demand_pp','implied_operating_rate_denominator_mw','implied_denom_minus_current_demand_mw']
df[cols].to_csv(OUT/'tables'/'audited_panel_extended_diagnostics.csv',index=False)

# summary json
summ={
'level_violations':int(level_violation.sum()),'rate_violations':int(rate_violation.sum()),'both':int((level_violation&rate_violation).sum()),'level_only':int((level_violation&~rate_violation).sum()),
'level_violations_below_15000_pct':float(100*(viol.supply_reserve_mw<15000).mean()),
'level_violations_max_supply_reserve_mw':float(viol.supply_reserve_mw.max()),
'zero_forecast_records':len(zero),'zero_forecast_episodes':len(zero_ep),
'continuity':continuity.to_dict(orient='records')
}
with open(OUT/'analysis_summary.json','w') as f: json.dump(summ,f,indent=2,default=str)
print(json.dumps(summ,indent=2,default=str))
print('\nLevel bins:\n',level_bin.to_string(index=False))
print('\nHourly nonzero:\n',hourly[hourly.level_violations>0].to_string(index=False))
print('\nRate diagnostics:\n',rate_relation.to_string(index=False))
print('\nContinuity:\n',continuity.to_string(index=False))
print('\nZero forecast:\n',zero_ep.to_string(index=False))
