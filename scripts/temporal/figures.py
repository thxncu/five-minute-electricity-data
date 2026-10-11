"""Generate publication figures for the temporal checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import HERE, OUTPUT_DIR

def save(fig,name,out):
    fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',facecolor='white',metadata={'Creator':None,'Producer':None,'CreationDate':None,'ModDate':None})
    fig.savefig(out/f'{name}.eps',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    out=OUTPUT_DIR/'figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    f=pd.read_csv(OUTPUT_DIR/'markov_and_exact_window_comparison.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,4.5),constrained_layout=True)
    for ax,mission,title in zip(axes[0],['cadence_delivery','full_service_strict'],['(a) Cadence windows','(b) Strict full-service windows']):
        sub=f[(f.mission_code==mission)&(f.period=='pooled_24_months')]
        for y,label,c,marker,ls in [('unavailability_pct','Observed','#1f4e79','o','-'),
                                  ('markov_unavailability_pct','Markov reference','#b45309','s','--'),
                                  ('iid_unavailability_pct','Independent reference','#616161','^',':')]:
            ax.plot(sub.window_minutes,sub[y],label=label,color=c,marker=marker,ms=3,linewidth=1.2,linestyle=ls)
        ax.set(xlabel='Required history (min)',ylabel='Unavailable windows (%)',title=title)
        ax.set_xticks([5,30,60,120,240]);ax.grid(axis='y',color='#e0e0e0',linewidth=.5)
    axes[0,0].legend(loc='upper left',frameon=False)
    runs=pd.read_csv(OUTPUT_DIR/'markov_run_length_diagnostics.csv')
    sub=runs[(runs.mission_code=='cadence_delivery')&(runs.run_state=='bad')&(runs.minimum_run_slots<=12)]
    positive=sub.observed_survival>0
    ax=axes[1,0]
    ax.plot(sub.loc[positive,'minimum_run_slots']*5,sub.loc[positive,'observed_survival'],color='#1f4e79',marker='o',label='Observed')
    ax.plot(sub.minimum_run_slots*5,sub.geometric_survival,color='#b45309',marker='s',linestyle='--',label='Markov geometric')
    ax.set_yscale('log');ax.set_ylim(1e-7,1.3);ax.set_xticks([5,10,15,30,60])
    ax.set(xlabel='Missing-episode duration threshold (min)',ylabel='Episode survival probability',title='(c) Cadence missing-episode tail')
    ax.grid(axis='y',color='#e0e0e0',linewidth=.5);ax.legend(frameon=False,loc='upper right')
    hold=pd.read_csv(OUTPUT_DIR/'markov_chronological_holdout.csv')
    sub=hold[(hold.mission_code=='cadence_delivery')&(hold.window_minutes<=60)]
    ax=axes[1,1]
    ax.plot(sub.window_minutes,sub.test_actual_unavailability_pct,color='#1f4e79',marker='o',label='Jan–Jun 2026 observed')
    ax.plot(sub.window_minutes,sub.train_fitted_markov_prediction_pct,color='#b45309',marker='s',linestyle='--',label='Jul 2024–Dec 2025 fit')
    ax.set(xlabel='Required history (min)',ylabel='Unavailable windows (%)',title='(d) Chronological cadence check')
    ax.set_xticks([5,10,15,30,60]);ax.grid(axis='y',color='#e0e0e0',linewidth=.5);ax.legend(frameon=False,loc='upper left')
    save(fig,'Temporal_Markov_Diagnostics',out)
    ctx=pd.read_csv(OUTPUT_DIR/'multichannel_change_june30_context.csv',parse_dates=['timestamp'])
    ctx=ctx[(ctx.timestamp>='2026-06-30 15:30')&(ctx.timestamp<='2026-06-30 18:00')]
    summary=json.loads((OUTPUT_DIR/'transition_summary.json').read_text())
    fig,axes_grid=plt.subplots(2,2,figsize=(10,4.5),sharex=True,constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.06,w_pad=0.06,hspace=0.08,wspace=0.06)
    axes=axes_grid.ravel()
    for ax,field,label in zip(axes[:3],['supply_capacity_mw','current_demand_mw','operating_reserve_mw'],['(a) Supply capacity','(b) Current demand','(c) Operating reserve']):
        ax.plot(ctx.timestamp,ctx[field]/1000,color='#1f4e79',marker='o',ms=2.5,linewidth=1)
        ax.set_ylabel('GW');ax.set_title(label,loc='left',pad=6);ax.grid(axis='y',color='#e0e0e0',linewidth=.5)
    ax=axes[3]
    ax.plot(ctx.timestamp,ctx.change_score,color='#1f4e79',marker='o',ms=2.5,linewidth=1,label='Change score')
    ax.axhline(summary['primary_score_threshold'],color='#b45309',linestyle='--',linewidth=1,label='99.9th-percentile cutoff')
    ax.set_ylabel('Change score');ax.set_title('(d) Change screen',loc='left',pad=6);ax.legend(loc='upper left',frameon=False)
    axes[0].set_yticks(np.arange(85,103,2.5))
    axes[3].set_yticks(np.arange(0,18,2.5))
    for ax in axes:
        ax.axvline(pd.Timestamp('2026-06-30 16:35'),color='#a61d24',linewidth=.8,linestyle=':')
        ax.axvline(pd.Timestamp('2026-06-30 16:45'),color='#a61d24',linewidth=.8,linestyle=':')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.supxlabel('2026-06-30, Korea Standard Time (KST; UTC+09:00)',fontsize=9)
    save(fig,'Temporal_Change_Screen_June30',out)

if __name__=='__main__':
    main()
