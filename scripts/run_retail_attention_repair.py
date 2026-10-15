"""Follow-up attention-mask repair, from the same SFT start and sampling budget.

Run after the five-arm scheduler exits. Preserve its frozen selection; this
registered follow-up has its own selection and test files.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import run_retail_grpo_repair as experiment


def main():
    root,out,art=experiment.ROOT,experiment.OUT,experiment.ART
    read,write=experiment.read,experiment.write
    baseline_config=read(out/'sft.json')
    config=experiment.clone(read(out/'progress.json'),
        art/'qwen25-3b-experiment-fast/checkpoints/0001','qwen25-3b-repair-attention')
    config['art']['packed_attention']=True
    if not read(out/'attention_gpu_probe.json')['passed']:
        raise RuntimeError('GPU correctness probe must pass before follow-up training')
    write(art/config['art']['name']/'attention_audit_request.json',
          {'descriptor':str(art/'qwen25-3b-repair-outcome/first_training_batch/descriptor.json')})
    folder=experiment.phase('attention',config,'grpo',['--steps','24','--workers','4',
                             '--task-ids','experiments/retail-grpo-repair/rl_tasks.json'])
    summaries=[read(p) for p in folder.glob('dev-round-*.json') if read(p)['round']>0]
    best=max(summaries,key=lambda s:(s['successes'],-s['budget_exhausted'],-s['repeated_calls'],-s['policy_step']))
    candidate=experiment.clone(config,art/config['art']['name']/'checkpoints'/f"{best['policy_step']:04d}",
                               'qwen25-3b-repair-attention-selected')
    candidate['seed']=43
    second=experiment.phase('attention-dev43',candidate,'evaluate',
                            ['--split','dev','--limit','15','--workers','4'])
    score42=experiment.score(experiment.valid_rows(folder,round_id=best['round']))
    score43=experiment.score(experiment.valid_rows(second))
    combined=[a+b for a,b in zip(score42,score43)]
    runs=read(out/'runs.json')
    sft_scores=[experiment.score(experiment.valid_rows(root/runs[f'sft-dev{seed}'])) for seed in (42,43)]
    sft_combined=[a+b for a,b in zip(*sft_scores)]
    selected={'arm':'attention','config':candidate,'step':best['policy_step'],'round':best['round'],
              'seed42_score':score42,'seed43_score':score43,'combined_score':combined,
              'training_folder':str(folder.relative_to(root))}
    decision={'selected':selected,'candidates':[selected],'sft_combined_score':sft_combined,
              'selection_data':'dev only; seeds 42 and 43; separate registered attention follow-up',
              'frozen_at':datetime.now(timezone.utc).isoformat()}
    destination=out/'selection-attention.json'
    if destination.exists():
        if read(destination)['selected']!=json.loads(json.dumps(selected)):
            raise RuntimeError('Refusing to change frozen attention follow-up selection')
    else:
        write(destination,decision)
    experiment.event('attention-selection','frozen',step=best['policy_step'],
                     rl_dev_score=combined,sft_dev_score=sft_combined)
    if combined[0]<=sft_combined[0]:
        experiment.event('attention-selection','no_dev_gain',reason='Follow-up did not pass registered development gate')
        return
    for seed in (42,43):
        for label,source in [('sft',baseline_config),('attention',candidate)]:
            test_config=json.loads(json.dumps(source));test_config['seed']=seed
            experiment.phase(f'{label}-test{seed}',test_config,'evaluate',
                             ['--split','test','--limit','40','--workers','4'])
    experiment.event('attention-experiment','complete')


if __name__=='__main__':
    main()
