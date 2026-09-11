#!/usr/bin/env python3
"""Candidate-blind MAS-yaw direction score with a frozen real-video adapter."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np

def read(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def yaw(row: dict[str, Any]) -> float | None:
    value = row.get("action_trajectory") or row.get("predicted_action_trajectory")
    if value is None: return None
    arr = np.asarray(value, dtype=object)
    if arr.dtype != object and arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] >= 3:
        return float(arr[0, 2, -1])
    vals = []
    for item in arr.tolist(): vals.append(item["pose"] if isinstance(item, dict) else item)
    return float(np.asarray(vals, dtype=float)[-1, 2])

def visual(row: dict[str, Any]) -> float | None:
    values = []
    for interval in row.get("flow_structure", {}).get("rows", []):
        value = interval.get("horizontal_flow_center")
        if value is not None and np.isfinite(value): values.append(float(value))
    return float(np.median(values)) if values else None

def wilson(h: int, n: int):
    if n == 0: return None
    z=1.959963984540054; p=h/n; den=1+z*z/n; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [(p+z*z/(2*n)-half)/den,(p+z*z/(2*n)+half)/den]

def fit_orientation(manifest, flows, action_threshold, visual_threshold):
    mm={(str(r.get("source_key")),str(r.get("branch_role", "logged_gt"))):r for r in manifest}
    pairs=[]
    for r in flows:
        source=mm.get((str(r.get("source_key")),str(r.get("branch_role","logged_gt")))) or mm.get((str(r.get("source_key")),"logged_gt"))
        if source is None: continue
        a=yaw(source); v=visual(r)
        if a is not None and v is not None and abs(a)>=action_threshold and abs(v)>=visual_threshold:
            pairs.append((np.sign(a),np.sign(v)))
    plus=sum(x==y for x,y in pairs); minus=sum(x==-y for x,y in pairs)
    orientation=1 if plus>=minus else -1
    hits=sum(orientation*x==y for x,y in pairs)
    return {"orientation": orientation, "n": len(pairs), "hits": hits, "accuracy": hits/len(pairs) if pairs else None, "ci95": wilson(hits,len(pairs)), "action_threshold": action_threshold, "visual_threshold": visual_threshold}

def score(manifest, flows, orientation, action_threshold, visual_threshold, action_sign=1, visual_sign=1, draws=20000):
    mm={(str(r.get("source_key")),str(r.get("branch_role"))):r for r in manifest}
    rows=[]
    for r in flows:
        source=mm.get((str(r.get("source_key")),str(r.get("branch_role"))))
        if source is None: continue
        a=yaw(source); v=visual(r)
        ok=a is not None and v is not None and abs(a)>=action_threshold and abs(v)>=visual_threshold
        rows.append({"source_key":str(r.get("source_key")),"branch_role":r.get("branch_role"),"action_yaw":a,"visual_center":v,"available":bool(ok),"hit":bool(ok and np.sign(visual_sign*orientation*v)==np.sign(action_sign*a))})
    grouped={}
    for r in rows: grouped.setdefault(r["source_key"],{})[str(r["branch_role"])] = r
    pair_rows=[]
    for source,b in grouped.items():
        usable=[x for x in b.values() if x["available"]]
        # A twin is available only when both symmetric branches are measurable.
        if "left" in b and "right" in b and b["left"]["available"] and b["right"]["available"]:
            pair_rows.append({"source_key":source,"hits":int(b["left"]["hit"])+int(b["right"]["hit"]),"n":2})
    rng=np.random.default_rng(20260912)
    means=[]
    if pair_rows:
        for _ in range(draws):
            sample=rng.integers(0,len(pair_rows),len(pair_rows)); means.append(np.mean([pair_rows[i]["hits"]/pair_rows[i]["n"] for i in sample]))
    hits=sum(r["hits"] for r in pair_rows); n=sum(r["n"] for r in pair_rows)
    return {"branch_count":len(rows),"branch_available":sum(r["available"] for r in rows),"branch_coverage":sum(r["available"] for r in rows)/len(rows) if rows else 0.0,"pair_count":len(grouped),"pair_available":len(pair_rows),"pair_coverage":len(pair_rows)/len(grouped) if grouped else 0.0,"direction_hits":hits,"direction_n":n,"direction_accuracy":hits/n if n else None,"direction_ci95":wilson(hits,n),"source_bootstrap_ci95":list(np.quantile(means,[.025,.975])) if means else None,"pairs":pair_rows,"branches":rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--calibration-manifest',type=Path,required=True); ap.add_argument('--calibration-flows',type=Path,required=True); ap.add_argument('--confirmation-manifest',type=Path,required=True); ap.add_argument('--confirmation-flows',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--action-threshold',type=float,default=5e-4); ap.add_argument('--visual-threshold',type=float,default=1e-4)
    a=ap.parse_args(); cal=fit_orientation(read(a.calibration_manifest),read(a.calibration_flows),a.action_threshold,a.visual_threshold); normal=score(read(a.confirmation_manifest),read(a.confirmation_flows),cal['orientation'],a.action_threshold,a.visual_threshold); reversed_=score(read(a.confirmation_manifest),read(a.confirmation_flows),cal['orientation'],a.action_threshold,a.visual_threshold,action_sign=-1); zero=score(read(a.confirmation_manifest),read(a.confirmation_flows),cal['orientation'],float('inf'),a.visual_threshold)
    out={'protocol':'iac-mas-yaw-direction-v1','metric_id':'MAS','metric_definition':'single_branch_visual_yaw_structure_aligned_with_native_action_direction','candidate_blind':True,'calibration':cal,'confirmation':{'normal':normal,'reversed_action':reversed_,'zero_action':zero},'formal_metric_eligible':bool(normal['pair_coverage']>=.9 and normal['source_bootstrap_ci95'] and normal['source_bootstrap_ci95'][0]>=.75),'claim_boundary':'directional structural alignment only; no metric distance, speed, lateral displacement, or future-to-action mediation claim'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,ensure_ascii=True,default=lambda value: value.item() if hasattr(value, "item") else str(value)))
    print(json.dumps({'calibration':cal,'normal':{k:normal[k] for k in ('pair_coverage','direction_accuracy','direction_ci95','source_bootstrap_ci95')},'formal_metric_eligible':out['formal_metric_eligible']}))
if __name__=='__main__': main()
