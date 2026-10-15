"""Read compact progress without loading models or printing dialogue/API content."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/retail-grpo-repair'


def records(path):
    if not path.exists():
        return []
    content = path.read_text()
    lines = content.splitlines()
    if content and not content.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def main():
    status = json.loads((OUT / 'status.json').read_text())
    runs = json.loads((OUT / 'runs.json').read_text())
    result = {'stage': status['stage'], 'state': status['state'], 'stage_time': status['time']}
    if status['stage'] in runs:
        folder = ROOT / runs[status['stage']]
        updates = records(folder / 'updates.jsonl')
        rows = records(folder / 'traces.jsonl')
        result.update(groups=len(updates), policy_updates=sum(row['updated'] for row in updates))
        evaluations = {}
        for row in rows:
            if row.get('error') or row['split'] == 'train':
                continue
            key = f"{row['split']}/seed{row['seed']}/round{row.get('evaluation_round', -1)}"
            evaluations.setdefault(key, {})[row['task_id']] = row['success']
        result['evaluations'] = {key: {'completed': len(values), 'successes': sum(values.values())}
                                 for key, values in evaluations.items()}
        result['infrastructure_failures'] = sum(bool(row.get('error')) for row in rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
