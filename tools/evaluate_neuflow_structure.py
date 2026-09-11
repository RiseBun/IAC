#!/usr/bin/env python3
"""Evaluate NeuFlow-only temporal structure on pure-speed twins.

This deliberately avoids depth and trajectory reconstruction: it is a flow
backbone A/B against the frozen RAFT structural descriptor.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np

def _load_model(root: Path, checkpoint: Path, device: str):
    import torch
    sys.path.insert(0, str(root))
    from NeuFlow.backbone_v7 import ConvBlock
    from NeuFlow.neuflow import NeuFlow
    model = NeuFlow().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    def fuse(conv, bn):
        fused = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)
        w = conv.weight.clone().view(conv.out_channels, -1)
        wb = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused.weight.copy_(torch.mm(wb, w).view(fused.weight.shape))
        b = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        bb = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.copy_(torch.mm(wb, b.reshape(-1, 1)).reshape(-1) + bb)
        return fused
    for module in model.modules():
        if type(module) is ConvBlock:
            module.conv1 = fuse(module.conv1, module.norm1); module.conv2 = fuse(module.conv2, module.norm2)
            delattr(module, "norm1"); delattr(module, "norm2"); module.forward = module.forward_fuse
    model.eval().half(); model.init_bhwd(1, 432, 768, device)
    return model, torch

def _flow(model, torch, first: str, second: str, device: str):
    a = cv2.imread(first, cv2.IMREAD_COLOR); b = cv2.imread(second, cv2.IMREAD_COLOR)
    if a is None or b is None: return None
    a = cv2.resize(a, (768, 432), interpolation=cv2.INTER_AREA); b = cv2.resize(b, (768, 432), interpolation=cv2.INTER_AREA)
    ta = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(device).half(); tb = torch.from_numpy(b).permute(2,0,1).unsqueeze(0).to(device).half()
    with torch.inference_mode():
        return model(ta, tb)[-1][0].float().permute(1,2,0).cpu().numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--neuflow-root',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--limit',type=int)
    a=ap.parse_args(); rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]; model,torch=_load_model(a.neuflow_root,a.checkpoint,a.device)
    by={}
    for r in rows: by.setdefault(r['source_key'],{})[r.get('speed_role')]=r
    sources=sorted(by)[:a.limit] if a.limit else sorted(by)
    branches=[]
    for i,s in enumerate(sources):
        for role in ('fast','slow'):
            r=by[s].get(role)
            if r is None: continue
            vals=[]
            for p,q in zip(r['future_frame_paths'][:-1],r['future_frame_paths'][1:]):
                flow=_flow(model,torch,p,q,a.device)
                if flow is None: continue
                h,w=flow.shape[:2]; roi=np.zeros((h,w),bool); roi[int(.50*h):int(.96*h),int(.10*w):int(.90*w)]=True
                mag=np.linalg.norm(flow,axis=-1); vals.append(float(np.median(mag[roi])))
            branches.append({'source_key':s,'speed_role':role,'interval_values':vals,'aggregate':float(np.mean(vals)) if vals else None})
        if (i+1)%5==0: print(f'{i+1}/{len(sources)}',flush=True)
    pairs=[]
    bmap={(r['source_key'],r['speed_role']):r for r in branches}
    for s in sources:
        f=bmap.get((s,'fast')); sl=bmap.get((s,'slow'))
        if f and sl and f['aggregate'] is not None and sl['aggregate'] is not None:
            pairs.append({'source_key':s,'fast':f['aggregate'],'slow':sl['aggregate'],'fast_wins':f['aggregate']>sl['aggregate']})
    out={'protocol':'iac-neuflow-structure-pilot-v1','descriptor':'median_flow_magnitude_px','branch_count':len(branches),'pair_count':len(pairs),'pair_direction_accuracy':float(np.mean([p['fast_wins'] for p in pairs])) if pairs else None,'branches':branches,'pairs':pairs,'formal_metric_eligible':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ('branch_count','pair_count','pair_direction_accuracy')}))
if __name__=='__main__': main()
