#!/usr/bin/env python3
"""Adapt WorldDrive command-0/command-2 outputs to the IAC pair schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def _benchmark_v3_future(row: dict) -> tuple[list, list, list]:
    future = list(row.get('future_frame_paths') or row.get('future_images') or [])
    action = list(row.get('action_trajectory') or [])
    times = list(row.get('future_timestamps') or row.get('future_times_s') or [])
    if len(future) == len(action) == len(times) == 8:
        selected = [1, 3, 5, 7]
        future = [future[index] for index in selected]
        action = [action[index] for index in selected]
        times = [times[index] for index in selected]
    if len(future) != 4 or len(action) != 4 or len(times) != 4:
        raise ValueError(f"{row.get('sample_id')}: expected aligned 4- or 8-step future/action data")
    if any(abs(float(value) - target) > 0.02 for value, target in zip(times, (1.0, 2.0, 3.0, 4.0))):
        raise ValueError(f"{row.get('sample_id')}: future timeline is not Benchmark-v3 compatible: {times}")
    return future, action, [1.0, 2.0, 3.0, 4.0]


def adapt_rows(rows: list[dict], *, allow_native_single: bool = False) -> list[dict]:
    groups={}
    for r in rows:
        command_index = r.get('command_index')
        if command_index is None:
            try: command_index = int(str(r.get('branch_id','')).split('_')[-1])
            except ValueError: command_index = -1
        if int(command_index) in (0,2):
            role='left' if int(command_index)==0 else 'right'
        elif allow_native_single and str(r.get('branch_id') or 'native') == 'native':
            role='native'
        else:
            continue
        future, action, future_times = _benchmark_v3_future(r)
        out=dict(r); out['branch_role']=role; out['counterfactual_group_id']=str(r['source_key'])
        out['sample_id']=str(r['sample_id'])
        out['future_frame_paths']=future; out['future_images']=future; out['action_trajectory']=action
        out['frame_paths']=list(r['history_frame_paths'])+future
        out['history_count']=len(r['history_frame_paths']); out['future_count']=len(future)
        out['history_times_s']=[-1.5,-1.0,-0.5,0.0]; out['future_times_s']=future_times
        out['intrinsics_source_size']=[1024,512]
        out['candidate_bank_used_by_measurement']=False; out['metric_reconstruction_used']=False
        # The native action head is an input to the generated branch, never a
        # logged-GT reference.  Explicitly clear any legacy value inherited
        # from the producer manifest so downstream validators fail closed.
        out['gt_candidate_id']=None
        out['action_trajectory_source']='wam_action_head'
        out['future_images_source']='wam_generated'
        metadata=dict(out.get('metadata') or {})
        metadata['benchmark_v3_adapter']={
            'source_times_s': list(r.get('future_timestamps') or r.get('future_times_s') or []),
            'selected_indices': [1,3,5,7] if len(r.get('future_frame_paths') or []) == 8 else [0,1,2,3],
            'selection': 'exact_timestamp_no_interpolation',
        }
        out['metadata']=metadata
        groups.setdefault(str(r['source_key']),{})[role]=out
    output=[]
    for _, branches in sorted(groups.items()):
        if 'native' in branches:
            output.append(branches['native'])
        elif 'left' in branches and 'right' in branches:
            output.extend([branches['left'], branches['right']])
    return output


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-native-single',action='store_true',help='Allow one native action/future row per source for AS; paired mode remains the default.')
    a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    result = adapt_rows(rows, allow_native_single=a.allow_native_single)
    a.output.write_text('\n'.join(json.dumps(r) for r in result)+'\n')
    print(json.dumps({'sources':len({r['source_key'] for r in result}),'branches':len(result),'output':str(a.output)}))
if __name__=='__main__': main()
