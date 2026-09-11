#!/usr/bin/env python3
"""Recompute the frozen RCS-yaw structural response and source bootstrap CI."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np

def read(path: Path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def endpoint_yaw(row: dict[str, Any]) -> float:
    value=row.get('action_trajectory') or row.get('predicted_action_trajectory'); arr=np.asarray(value,dtype=object)
    if arr.dtype != object and arr.ndim==3 and arr.shape[0]==1: return float(arr[0,2,-1])
    poses=[x['pose'] if isinstance(x,dict) else x for x in arr.tolist()]; return float(np.asarray(poses,dtype=float)[-1,2])
def flow_values(row):
    out={}
    for q in row.get('flow_structure',{}).get('rows',[]):
        v=q.get('horizontal_flow_center')
        if v is not None and np.isfinite(v): out[int(q.get('interval_index',len(out)))]=float(v)
    return out
def wilson(h,n):
    if not n:return None
    z=1.959963984540054;p=h/n; den=1+z*z/n; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return [(p+z*z/(2*n)-half)/den,(p+z*z/(2*n)+half)/den]
def evaluate(manifest, flows, orientation=1, action_threshold=.01, visual_threshold=1e-4, action_sign=1, draws=20000):
    mm={(str(r.get('source_key')),str(r.get('branch_role'))):r for r in manifest}; ff={(str(r.get('source_key')),str(r.get('branch_role'))):r for r in flows}
    sources=sorted({s for s,r in ff if r=='left'} & {s for s,r in ff if r=='right'}); pairs=[]
    for source in sources:
        left,right=ff[(source,'left')],ff[(source,'right')]; lv,rv=flow_values(left),flow_values(right); common=sorted(set(lv)&set(rv))
        if len(common)<2: continue
        delta=float(np.median([lv[i]-rv[i] for i in common])); action_delta=endpoint_yaw(mm[(source,'left')])-endpoint_yaw(mm[(source,'right')]); comparable=abs(action_delta)>=action_threshold and abs(delta)>=visual_threshold
        pairs.append({'source_key':source,'common_intervals':common,'flow_delta':delta,'action_delta':action_delta,'comparable':bool(comparable),'normal_hit':bool(comparable and np.sign(orientation*delta)==np.sign(action_delta)),'reversed_hit':bool(comparable and np.sign(orientation*delta)==np.sign(-action_delta))})
    eligible=[p for p in pairs if p['comparable']]; hits=sum(p['normal_hit'] for p in eligible); rng=np.random.default_rng(20260912); means=[]
    for _ in range(draws):
        sample=rng.integers(0,len(eligible),len(eligible)); means.append(np.mean([int(eligible[i]['normal_hit']) for i in sample])) if eligible else None
    return {'pair_count':len(pairs),'pair_coverage':len(pairs)/len(sources) if sources else 0.0,'direction_pairs':len(eligible),'direction_hits':hits,'direction_accuracy':hits/len(eligible) if eligible else None,'direction_ci95':wilson(hits,len(eligible)),'source_bootstrap_ci95':list(np.quantile(means,[.025,.975])) if means else None,'reversed_direction_accuracy':sum(p['reversed_hit'] for p in eligible)/len(eligible) if eligible else None,'pairs':pairs}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--flows',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--model',required=True); a=ap.parse_args(); out={'protocol':'iac-rcs-yaw-direction-v1','metric_id':'RCS','metric_definition':'same-source differential horizontal-flow structure aligned with native yaw intervention','candidate_blind':True,'adapter':{'orientation':1,'action_yaw_delta_threshold_rad':.01,'flow_delta_threshold':1e-4,'calibration_domain':'logged_realized_navsim_future','calibration_source':'configs/mas_yaw_v1.json'},'model':a.model,'normal':evaluate(read(a.manifest),read(a.flows),action_sign=1),'controls':{'reversed_action':'reported in normal.reversed_direction_accuracy'},'claim_boundary':'directional counterfactual response only; not metric distance or future-to-action mediation'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
if __name__=='__main__': main()
