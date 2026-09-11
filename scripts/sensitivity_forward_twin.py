#!/usr/bin/env python3
"""Sensitivity curve for twin differential direction vector threshold."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from iac_new.scoring import polygon_mask
from iac_new.visual_consistency import score_twin_differential_consistency, trajectory_conditioned_flow

def traj(v, n):
    if len(v) == n: return v
    if len(v) == 2*n: return v[[1,3,5,7]]
    return v[np.linspace(0,len(v)-1,n).round().astype(int)]

def run(root, threshold):
    metas=[]
    for p in sorted(root.glob('*.json')):
        if p.name == 'manifest.json': continue
        m=json.loads(p.read_text())
        if 'flow_path' in m: metas.append(m)
    by={}
    for m in metas: by.setdefault(str(m['source_key']),{})[str(m.get('branch_role'))]=m
    rows=[]
    for source,b in sorted(by.items()):
        if 'left' not in b or 'right' not in b: continue
        lm,rm=b['left'],b['right']; lf,rf=np.load(root/lm['flow_path']),np.load(root/rm['flow_path'])
        n=lf.shape[0]; lt,rt=traj(np.asarray(lm['trajectory']),n),traj(np.asarray(rm['trajectory']),n); shape=lf.shape[1:3]
        le,lg=trajectory_conditioned_flow(lt,intrinsics=np.asarray(lm['intrinsics']),camera_to_ego=np.asarray(lm['camera_to_ego']),frame_shape=shape)
        re,rg=trajectory_conditioned_flow(rt,intrinsics=np.asarray(rm['intrinsics']),camera_to_ego=np.asarray(rm['camera_to_ego']),frame_shape=shape)
        roi=polygon_mask(shape[0],shape[1],[[.08,.98],[.92,.98],[.63,.53],[.37,.53]])
        common=np.load(root/lm['valid_path']) & np.load(root/rm['valid_path']) & lg & rg & roi
        for control,el,er in [('normal',le,re),('reversed',re,le),('zero',re,re)]:
            s=score_twin_differential_consistency(lf,rf,el,er,valid_mask=common,min_vector_norm_px=threshold)
            rows.append({'control':control,**{k:s[k] for k in ('interval_coverage','median_residual_px','median_observed_delta_px','median_expected_delta_px','median_direction_cosine','median_direction_vector_fraction')}})
    out={}
    for c in ('normal','reversed','zero'):
        out[c]={}
        for k in ('interval_coverage','median_residual_px','median_observed_delta_px','median_expected_delta_px','median_direction_cosine','median_direction_vector_fraction'):
            vals=[r[k] for r in rows if r['control']==c and r[k] is not None]; out[c][k]=float(np.median(vals)) if vals else None
    gains=[r['median_observed_delta_px']/r['median_expected_delta_px'] for r in rows if r['control']=='normal' and r['median_expected_delta_px'] is not None and r['median_expected_delta_px']>0]
    out['normal_response_gain_median']=float(np.median(gains)) if gains else None
    return {'min_vector_norm_px':threshold,'controls':out}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--thresholds',type=float,nargs='+',default=[.05,.1,.25,.5,1.0]); a=p.parse_args()
    result={'protocol':'iac-twin-differential-forward-consistency-sensitivity-v1','curves':[run(a.root,t) for t in a.thresholds]}; a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
