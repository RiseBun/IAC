#!/usr/bin/env python3
"""Adapt WorldDrive command-0/command-2 outputs to the IAC pair schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def adapt_rows(rows: list[dict]) -> list[dict]:
    groups={}
    for r in rows:
        command_index = r.get('command_index')
        if command_index is None:
            try: command_index = int(str(r.get('branch_id','')).split('_')[-1])
            except ValueError: command_index = -1
        if int(command_index) not in (0,2): continue
        role='left' if int(command_index)==0 else 'right'
        out=dict(r); out['branch_role']=role; out['counterfactual_group_id']=str(r['source_key'])
        out['sample_id']=str(r['sample_id']); out['frame_paths']=list(r['history_frame_paths'])+list(r['future_frame_paths'])
        out['history_count']=len(r['history_frame_paths']); out['future_count']=len(r['future_frame_paths'])
        out['future_times_s']=list(r.get('future_timestamps', r.get('future_times_s'))); out['intrinsics_source_size']=[1024,512]
        out['candidate_bank_used_by_measurement']=False; out['metric_reconstruction_used']=False
        # The native action head is an input to the generated branch, never a
        # logged-GT reference.  Explicitly clear any legacy value inherited
        # from the producer manifest so downstream validators fail closed.
        out['gt_candidate_id']=None
        out['action_trajectory_source']='wam_action_head'
        out['future_images_source']='wam_generated'
        groups.setdefault(str(r['source_key']),{})[role]=out
    return [v[role] for _,v in sorted(groups.items()) if 'left' in v and 'right' in v for role in ('left','right')]


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    result = adapt_rows(rows)
    a.output.write_text('\n'.join(json.dumps(r) for r in result)+'\n')
    print(json.dumps({'sources':len(result)//2,'branches':len(result),'output':str(a.output)}))
if __name__=='__main__': main()
