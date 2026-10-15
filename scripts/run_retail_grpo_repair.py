"""Controlled GRPO arms, dev-only selection, then paired frozen tests."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/retail-grpo-repair'
TRACES = ROOT / 'traces/retail/grpo-repair'
ART = ROOT / '.art/service-agent-retail/models'
MARKERS = ROOT / '.art/retail-state/service-agent-retail'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def event(stage, state, **kwargs):
    value = dict(time=datetime.now(timezone.utc).isoformat(), stage=stage, state=state, **kwargs)
    with (OUT / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False)+'\n')
    write(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def clone(config, source, destination_name):
    config = json.loads(json.dumps(config))
    config['art']['name'] = destination_name
    target = ART / destination_name / 'checkpoints' / source.name
    if not target.exists():
        # Evaluation clones need the default adapter/tokenizer only; optimizer
        # states and frozen reference adapters are not inference inputs.
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file() and path.name != 'retail_optimizer.pt':
                shutil.copy2(path, target / path.name)
        shutil.copy2(MARKERS / 'qwen25-3b-experiment-fast.json', MARKERS / (destination_name+'.json'))
        write(target.parent.parent / 'fork.json', {
            'source': str(source.relative_to(ROOT)),
            'sha256': hashlib.sha256((source/'adapter_model.safetensors').read_bytes()).hexdigest()})
    elif (target/'adapter_model.safetensors').read_bytes() != (source/'adapter_model.safetensors').read_bytes():
        raise RuntimeError('Refusing to overwrite a model fork with a different source')
    return config


def phase(name, config, stage, arguments):
    config_path = OUT / f'{name}.json'
    if config_path.exists() and read(config_path) != config:
        raise RuntimeError('Phase config changed; record a new experiment, not an overwrite')
    write(config_path, config)
    runs = read(OUT/'runs.json') if (OUT/'runs.json').exists() else {}
    existing = runs.get(name)
    if existing and (ROOT/existing/'phase-complete.json').exists():
        return ROOT/existing
    attempts = 3 if stage == 'evaluate' else 1
    for attempt in range(attempts):
        before = set(TRACES.glob(f'*_art_{stage}_*'))
        command = ['bash', 'scripts/retail.sh', stage, *arguments]
        if existing:
            command += ['--resume', existing]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log_path = TRACES/'logs'/f'{name}-{stamp}.log'
        event(name, 'started', command=command, config=str(config_path.relative_to(ROOT)),
              log=str(log_path.relative_to(ROOT)))
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONUNBUFFERED': '1',
               'RETAIL_CONFIG': str(config_path)}
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                new = set(TRACES.glob(f'*_art_{stage}_*')) - before
                if not existing and new:
                    if len(new) != 1:
                        process.terminate()
                        raise RuntimeError('Ambiguous experiment folder')
                    existing = str(new.pop().relative_to(ROOT))
                    runs[name] = existing
                    write(OUT/'runs.json', runs)
                gpu = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if gpu.returncode == 0:
                    with (OUT/'gpu.csv').open('a') as samples:
                        samples.write(f'{datetime.now(timezone.utc).isoformat()},{name},{gpu.stdout.strip()}\n')
                time.sleep(5)
        event(name, 'finished' if process.returncode == 0 else 'failed',
              returncode=process.returncode, folder=existing)
        if process.returncode == 0:
            write(ROOT/existing/'phase-complete.json', {'phase': name, 'log': str(log_path.relative_to(ROOT))})
            return ROOT/existing
    raise RuntimeError(f'{name} failed; remaining phases are stopped')


def valid_rows(folder, split='dev', round_id=None):
    rows = [json.loads(line) for line in (folder/'traces.jsonl').read_text().splitlines()]
    chosen = {}
    for row in rows:
        if row['split'] == split and not row.get('error') and (round_id is None or row.get('evaluation_round') == round_id):
            chosen[row['task_id']] = row
    expected = set(read(ROOT/'data/tau2/splits.json')[split])
    if set(chosen) != expected:
        raise RuntimeError(f'Incomplete {split} task coverage: {folder}')
    return list(chosen.values())


def score(rows):
    from mini_customer_agent.retail.rewards import process_signals
    signals = [process_signals(row) for row in rows]
    return (sum(row['success'] for row in rows),
            -sum(s['budget_exhausted'] for s in signals),
            -sum(s['repeated_calls'] for s in signals))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dev-only',action='store_true',help='Freeze development results without launching tests')
    args=parser.parse_args()
    config = read(OUT/'sft.json')
    baseline = phase('sft-dev42', config, 'evaluate', ['--split', 'dev', '--limit', '15', '--workers', '4'])
    baseline42 = valid_rows(baseline)
    candidates = []
    # Exercise the new reference-adapter path first; statistical arm definitions
    # and the task/seed schedule are independent of execution order.
    for arm in ['kl', 'outcome', 'consistent', 'shaped', 'progress']:
        arm_config = clone(config, ART/'qwen25-3b-experiment-fast/checkpoints/0001', 'qwen25-3b-repair-'+arm)
        arm_config['art'].update(learning_rate=2e-6, persist_optimizer=True, eval_every=8,
                                 fixed_group_seeds=True, kl_beta=0 if arm == 'outcome' else .02,
                                 reward={'mode': ('business_progress' if arm == 'progress' else
                                                  'bounded_process' if arm in {'shaped', 'consistent'} else 'outcome')})
        if arm in {'consistent', 'progress'}:
            arm_config['art'].update(load_in_4bit=False, stable_logprobs=True, packed_positions=True)
        if arm != 'outcome':
            arm_config['art']['kl_reference'] = str(ART/'qwen25-3b-experiment-fast/checkpoints/0001')
        folder = phase(arm, arm_config, 'grpo', ['--steps', '24', '--workers', '4',
                       '--task-ids', 'experiments/retail-grpo-repair/rl_tasks.json'])
        summaries = [read(path) for path in sorted(folder.glob('dev-round-*.json')) if read(path)['round'] > 0]
        best = max(summaries, key=lambda s: (s['successes'], -s['budget_exhausted'], -s['repeated_calls'], -s['policy_step']))
        step = best['policy_step']
        candidate = clone(arm_config, ART/arm_config['art']['name']/'checkpoints'/f'{step:04d}',
                          'qwen25-3b-repair-'+arm+'-selected')
        candidate['seed'] = 43
        candidates.append({'arm': arm, 'config': candidate, 'step': step, 'round': best['round'],
                           'seed42_score': score(valid_rows(folder, round_id=best['round'])),
                           'training_folder': str(folder.relative_to(ROOT))})
    write(OUT/'dev_candidates.json', candidates)
    sft43 = json.loads(json.dumps(config))
    sft43['seed'] = 43
    folder = phase('sft-dev43', sft43, 'evaluate', ['--split', 'dev', '--limit', '15', '--workers', '4'])
    baseline43 = valid_rows(folder)
    for candidate in candidates:
        folder = phase(candidate['arm']+'-dev43', candidate['config'], 'evaluate',
                       ['--split', 'dev', '--limit', '15', '--workers', '4'])
        candidate['seed43_score'] = score(valid_rows(folder))
        candidate['combined_score'] = [a+b for a,b in zip(candidate['seed42_score'], candidate['seed43_score'])]
    selected = max(candidates, key=lambda c: (*c['combined_score'], -c['step']))
    decision = {'selected': selected, 'candidates': candidates,
                'sft_combined_score': [a+b for a,b in zip(score(baseline42), score(baseline43))],
                'selection_data': 'dev only; seeds 42 and 43',
                'frozen_at': datetime.now(timezone.utc).isoformat()}
    if (OUT/'selection.json').exists():
        frozen = read(OUT/'selection.json')
        if frozen['selected'] != json.loads(json.dumps(decision['selected'])):
            raise RuntimeError('Frozen selection differs; test-driven reselection is prohibited')
    else:
        write(OUT/'selection.json', decision)
    event('selection', 'frozen', arm=selected['arm'], step=selected['step'],
          rl_dev_score=selected['combined_score'], sft_dev_score=decision['sft_combined_score'])
    if args.dev_only:
        event('selection','development_complete',reason='Original five arms retained as diagnostic controls; attention repair has its own gate')
        return
    if selected['combined_score'][0] <= decision['sft_combined_score'][0]:
        event('selection', 'no_dev_gain', reason='Do not spend or select on test before a development improvement')
        return
    for seed in [42, 43]:
        for label, source in [('sft', config), ('selected', selected['config'])]:
            test_config = json.loads(json.dumps(source))
            test_config['seed'] = seed
            phase(f'{label}-test{seed}', test_config, 'evaluate',
                  ['--split', 'test', '--limit', '40', '--workers', '4'])
    event('experiment', 'complete')


if __name__ == '__main__':
    main()
