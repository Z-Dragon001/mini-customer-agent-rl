"""Audit completed real runs, reward integrity, held-out isolation and selection."""
import hashlib
import argparse
import json
import math
import os
import random
from pathlib import Path
from unittest.mock import patch

from mini_customer_agent.retail.rewards import reward_breakdown, group_is_informative, process_signals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/retail-grpo-repair'


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-only', action='store_true',
                        help='Audit completed arms before frozen test evaluation')
    parser.add_argument('--attention', action='store_true', help='Audit the separate attention follow-up')
    args = parser.parse_args()
    runs=read(OUT/'runs.json')
    splits=read(ROOT/'data/tau2/splits.json')
    checks=[]
    curriculum=read(OUT/'rl_tasks.json')
    random.Random(43).shuffle(curriculum)
    def check(name, condition):
        checks.append({'check':name,'passed':bool(condition)})
    check('disjoint train/dev/test tasks', all(not set(splits[a]) & set(splits[b])
          for a,b in (('train','dev'),('train','test'),('dev','test'))))
    check('curriculum contains training tasks only',set(curriculum)<=set(splits['train']))
    arms=('attention',) if args.attention else ('outcome','kl','shaped','consistent','progress')
    for arm in arms:
        folder=ROOT/runs[arm]
        spec=read(folder/'run.json')['config']['art']
        rows=records(folder/'traces.jsonl')
        updates=records(folder/'updates.jsonl')
        selected={(r['group_id'],r['rollout_index']):r for r in rows
                  if r['split']=='train' and not r.get('error')}
        check(arm+':24 contiguous groups',[r['iteration'] for r in updates]==list(range(24)))
        check(arm+':96 selected trajectories',set(selected)=={(g,i) for g in range(24) for i in range(4)})
        check(arm+':train-only tasks',all(r['task_id'] in splits['train'] for r in selected.values()))
        check(arm+':official score agrees with success',
              all(r['success']==(r['official_reward']==1.0) for r in selected.values()))
        if arm in {'progress','attention'}:
            os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
            from mini_customer_agent.retail.config import configure_tau
            from mini_customer_agent.retail.business_progress import calculate_business_progress
            configure_tau(read(folder/'run.json')['config'])
            from loguru import logger
            logger.disable('tau2')
            with patch('httpx.Client.send',side_effect=AssertionError('Audit attempted network access')):
                for (group,index),row in selected.items():
                    before=json.dumps(row['messages'],sort_keys=True)
                    expected_progress=calculate_business_progress(row)
                    check(f'{arm}:group{group}:trajectory{index}:offline progress',
                          row['business_progress']==expected_progress)
                    check(f'{arm}:group{group}:trajectory{index}:observations unchanged',
                          json.dumps(row['messages'],sort_keys=True)==before)
        for group in updates:
            trajectories=[selected[group['iteration'],i] for i in range(4)]
            check(f'{arm}:group{group["iteration"]}:curriculum',
                  group['task_id']==curriculum[group['iteration']]
                  and all(r['task_id']==group['task_id'] for r in trajectories))
            check(f'{arm}:group{group["iteration"]}:sampling checkpoint',
                  all(r['policy_step']==group['policy_step']-int(group['updated']) for r in trajectories))
            expected=[reward_breakdown(r,spec['reward']) for r in trajectories]
            check(f'{arm}:group{group["iteration"]}:reward',
                  all(a['training_reward']==b for a,b in zip(trajectories,expected))
                  and group['rewards']==[b['total'] for b in expected])
            check(f'{arm}:group{group["iteration"]}:update',
                  group['updated']==group_is_informative(group['rewards'],[r['success'] for r in trajectories],spec['reward'],
                      progresses=[r.get('business_progress',{}).get('score',0) for r in trajectories]))
            check(f'{arm}:group{group["iteration"]}:seed',
                  all(r['seed']==1042+group['iteration'] for r in trajectories))
        expected_step=1+sum(u['updated'] for u in updates)
        check(arm+':final checkpoint step',updates[-1]['policy_step']==expected_step)
        model=ROOT/spec['path']/spec['project']/'models'/spec['name']
        source=ROOT/'.art/service-agent-retail/models/qwen25-3b-experiment-fast/checkpoints/0001/adapter_model.safetensors'
        check(arm+':same SFT starting weights',
              (model/'checkpoints/0001/adapter_model.safetensors').read_bytes()==source.read_bytes())
        for step in range(2,expected_step+1):
            check(f'{arm}:optimizer{step}',(model/'checkpoints'/f'{step:04d}'/'retail_optimizer.pt').exists())
        for round_id in (8,16,24):
            dev={r['task_id']:r for r in rows if r['split']=='dev' and
                 r.get('evaluation_round')==round_id and not r.get('error')}
            check(f'{arm}:dev{round_id}',set(dev)==set(splits['dev']))
        metrics=records(folder/'grpo_batch_metrics.jsonl')
        check(arm+':finite updates',all(math.isfinite(v) for row in metrics for v in row['metrics'].values()
                                      if isinstance(v,(float,int))))
        if arm!='outcome':
            check(arm+':KL measured',bool(metrics) and all('kl_div' in m['metrics'] for m in metrics))
    selection_path=OUT/('selection-attention.json' if args.attention else 'selection.json')
    frozen=read(selection_path) if selection_path.exists() else None
    if frozen:
        def dev_score(folder, round_id=-1):
            rows={r['task_id']:r for r in records(folder/'traces.jsonl')
                  if r['split']=='dev' and not r.get('error')
                  and r.get('evaluation_round',-1)==round_id}
            check(str(folder.name)+f':selection coverage:{round_id}',set(rows)==set(splits['dev']))
            signals=[process_signals(row) for row in rows.values()]
            return [sum(r['success'] for r in rows.values()),
                    -sum(s['budget_exhausted'] for s in signals),
                    -sum(s['repeated_calls'] for s in signals)]
        sft_combined=[a+b for a,b in zip(dev_score(ROOT/runs['sft-dev42']),dev_score(ROOT/runs['sft-dev43']))]
        check('selection baseline scores match traces',sft_combined==frozen['sft_combined_score'])
        check('registered candidates recorded',{c['arm'] for c in frozen['candidates']}==set(arms)
              and len(frozen['candidates'])==len(arms))
        for candidate in frozen['candidates']:
            arm=candidate['arm']
            folder=ROOT/runs[arm]
            summaries=[read(p) for p in folder.glob('dev-round-*.json') if read(p)['round']>0]
            best=max(summaries,key=lambda s:(s['successes'],-s['budget_exhausted'],-s['repeated_calls'],-s['policy_step']))
            check(arm+':seed42 checkpoint selection',
                  (candidate['round'],candidate['step'])==(best['round'],best['policy_step']))
            score42=dev_score(folder,candidate['round'])
            score43=dev_score(ROOT/runs[arm+'-dev43'])
            check(arm+':combined selection score',candidate['seed42_score']==score42
                  and candidate['seed43_score']==score43
                  and candidate['combined_score']==[a+b for a,b in zip(score42,score43)])
            spec=read(folder/'run.json')['config']['art']
            source=ROOT/spec['path']/spec['project']/'models'/spec['name']/'checkpoints'/f"{candidate['step']:04d}"
            clone=ROOT/spec['path']/spec['project']/'models'/candidate['config']['art']['name']/'checkpoints'/source.name
            check(arm+':selected adapter matches checkpoint',
                  (source/'adapter_model.safetensors').read_bytes()==(clone/'adapter_model.safetensors').read_bytes())
        expected=max(frozen['candidates'],key=lambda c:(*c['combined_score'],-c['step']))
        check('final choice follows registered dev rule',expected==frozen['selected'])
        gate=frozen['selected']['combined_score'][0]>sft_combined[0]
        test_label='attention' if args.attention else 'selected'
        check('test gate respected',gate or not any(name.startswith(test_label+'-test') for name in runs))
        if not args.training_only:
            check('test follows a positive development gate',gate)
    for seed in (() if args.training_only else (42,43)):
        test_configs={}
        for stage in ('sft','attention' if args.attention else 'selected'):
            folder=ROOT/runs[f'{stage}-test{seed}']
            config=read(folder/'run.json')['config']
            test_configs[stage]=config
            check(f'{stage}:test{seed}:registered configuration',
                  config==read(OUT/f'{stage}-test{seed}.json'))
            rows=records(folder/'traces.jsonl')
            valid={r['task_id']:r for r in rows if not r.get('error')}
            check(f'{stage}:test{seed}:coverage',set(valid)==set(splits['test']))
            check(f'{stage}:test{seed}:split',all(r['split']=='test' and r['seed']==seed for r in valid.values()))
            check(f'{stage}:test{seed}:official score',
                  all(r['success']==(r['official_reward']==1.0) for r in valid.values()))
            if stage=='sft':
                check(f'sft:test{seed}:original checkpoint step',
                      all(r['policy_step']==1 for r in valid.values()))
                art=config['art']
                adapter=ROOT/art['path']/art['project']/'models'/art['name']/'checkpoints/0001/adapter_model.safetensors'
                original=ROOT/'.art/service-agent-retail/models/qwen25-3b-experiment-fast/checkpoints/0001/adapter_model.safetensors'
                check(f'sft:test{seed}:original checkpoint bytes',adapter.read_bytes()==original.read_bytes())
            if stage!='sft':
                check(f'selected:test{seed}:frozen checkpoint',all(r['policy_step']==frozen['selected']['step'] for r in valid.values()))
        baseline=test_configs['sft'];candidate=test_configs['attention' if args.attention else 'selected']
        check(f'test{seed}:matched environment and API configuration',
              {k:v for k,v in baseline.items() if k!='art'}==
              {k:v for k,v in candidate.items() if k!='art'})
        check(f'test{seed}:matched local inference configuration',all(
              baseline['art'].get(key)==candidate['art'].get(key) for key in
              ('base_model','max_seq_length','gpu_memory_utilization','temperature','enforce_eager')))
    for path,digest in read(OUT/'source_checkpoint.json').items():
        check('original SFT unchanged:'+Path(path).name,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    for path,digest in read(OUT/'numeric_reference_manifest.json').items():
        check('numerical replay input unchanged:'+Path(path).name,
              hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    numerical = read(ROOT/'.art/service-agent-retail/models/qwen25-3b-repair-consistent/numerical_audit.json')
    check('numerical replay before formal update', numerical['before_first_reported_update'])
    check('numerical replay finite', all(math.isfinite(value)
          for metrics in numerical['variants'].values() for value in metrics.values()))
    if args.attention:
        spec=read(ROOT/runs['attention']/'run.json')['config']['art']
        control=read(ROOT/runs['progress']/'run.json')['config']['art']
        check('attention is the only new training configuration',
              {k:v for k,v in spec.items() if k not in {'name','packed_attention'}}==
              {k:v for k,v in control.items() if k!='name'} and spec.get('packed_attention') is True)
        probe=read(OUT/'attention_gpu_probe.json')
        check('actual GPU forward/backward probe passed',probe['passed'])
        replay=read(ROOT/'.art/service-agent-retail/models/qwen25-3b-repair-attention/attention_audit.json')
        check('attention replay before formal update',replay['before_first_reported_update'])
        descriptor=read(Path(replay['source_descriptor']))
        check('attention replay covers all frozen sequences',len(replay['sequences'])==descriptor['num_sequences'])
        check('attention replay finite',all(math.isfinite(value) for sequence in replay['sequences']
              for metrics in sequence['variants'].values() for value in metrics.values()))
        resume_manifest=OUT/'attention_resume_manifest.json'
        if resume_manifest.exists():
            for path,digest in read(resume_manifest).items():
                if path.endswith('attention_audit.json'):
                    check('initial attention audit preserved across resume',
                          hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    result={'scope':'training' if args.training_only else 'training and frozen test',
            'passed':all(c['passed'] for c in checks),'checks':checks,
            'limits':['Does not assert a positive effect or full policy compliance',
                      'Remote user/judge aliases and generation are not fully deterministic']}
    suffix=('-attention' if args.attention else '')+('-training' if args.training_only else '')
    (OUT/f'audit{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),
                      'failed':[c for c in checks if not c['passed']]},indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
