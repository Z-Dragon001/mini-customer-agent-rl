"""Isolated episode workers: tau's RNG and judge globals are process-local."""
import traceback


def episode_job(job):
    from openai import OpenAI
    from .config import endpoint_kwargs, generation_kwargs
    from .rollout import run_episode
    config = job["config"]
    source = job["source"]
    connection = job.get("connection") or endpoint_kwargs(config[source])
    generation = job.get("generation") or generation_kwargs(config[source])
    model = job.get("model") or config[source]["model"]
    try:
        with OpenAI(**connection) as client:
            row, history = run_episode(config, job["task_id"], job["split"], client.chat.completions.create, model,
                               source=source, seed=job["seed"], tasks_sha256=job["tasks_sha256"], generation=generation)
            if (job['split'] == 'train' and
                    config.get('art', {}).get('reward', {}).get('mode') == 'business_progress'):
                from .business_progress import calculate_business_progress
                row['business_progress'] = calculate_business_progress(row)
            return row, history
    except Exception as exc:
        from .data import REVISION
        return {"task_id": job["task_id"], "split": job["split"], "source": source,
                "seed": job["seed"], "model": model, "tau_revision": REVISION,
                "tasks_sha256": job["tasks_sha256"], "success": False, "error": type(exc).__name__,
                "error_frames": [{"file": f.filename, "line": f.lineno, "function": f.name}
                                 for f in traceback.extract_tb(exc.__traceback__)]}, []
