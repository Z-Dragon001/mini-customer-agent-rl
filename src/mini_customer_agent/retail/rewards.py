"""Auditable process signals; training rewards never replace official scores."""
from __future__ import annotations

import json
import math

READ_TOOLS = {"get_user_details", "get_order_details", "get_product_details", "get_item_details",
              "list_all_product_types", "find_user_id_by_email", "find_user_id_by_name_zip"}
WRITE_TOOLS = {"cancel_pending_order", "modify_pending_order_address", "modify_pending_order_items",
               "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items",
               "exchange_delivered_order_items"}


def process_signals(row: dict) -> dict:
    """Count repeats of successful calls within an unchanged database epoch.

    A read after a successful mutation is not a duplicate. This is a process
    proxy, not a semantic proof of redundancy or a policy-compliance score.
    """
    pending, seen_reads, seen_writes = {}, set(), set()
    repeated = 0
    simulation = row.get("simulation", {})
    errors = {m.get("id"): bool(m.get("error")) for m in
              simulation.get("messages", simulation.get("partial_messages", []))
              if m.get("role") == "tool"}
    for message in row.get("messages", []):
        for call in message.get("tool_calls") or []:
            function = call["function"]
            try:
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                key = (function["name"], json.dumps(arguments, sort_keys=True, ensure_ascii=False))
            except (TypeError, ValueError):
                continue
            pending[call["id"]] = key
        if message.get("role") != "tool":
            continue
        key = pending.pop(message.get("tool_call_id"), None)
        if key is None:
            continue
        content = message.get("content") or ""
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (ValueError, TypeError):
            payload = None
        failed = errors.get(message.get("tool_call_id"), False) or (
            isinstance(payload, dict) and bool(payload.get("error"))) or str(content).startswith("Error:")
        seen = seen_reads if key[0] in READ_TOOLS else seen_writes
        if key in seen:
            repeated += 1
        if not failed:
            if key[0] in WRITE_TOOLS:
                seen_reads.clear()
                seen_writes.clear()
                seen_writes.add(key)
            elif key[0] in READ_TOOLS:
                seen_reads.add(key)
    return {"repeated_calls": repeated,
            "budget_exhausted": int(row.get("termination") in {"max_steps", "context_length_exceeded"}),
            "invalid_output": int(row.get("termination") == "invalid_model_output"),
            "tool_errors": int(row.get("tool_errors", 0))}


def reward_breakdown(row: dict, spec: dict | None = None) -> dict:
    if row.get("error"):
        raise ValueError("Infrastructure failures cannot be training samples")
    spec = spec or {"mode": "outcome"}
    mode = spec.get("mode", "outcome")
    if mode not in {"outcome", "bounded_process", "business_progress"}:
        raise ValueError(f"Unknown reward mode: {mode}")
    official = float(row["official_reward"])
    if official not in (0.0, 1.0):
        raise ValueError("Expected binary official task score")
    signals = process_signals(row)
    components = {}
    if mode in {"bounded_process", "business_progress"}:
        for key, default in [("repeated_calls", .02), ("tool_errors", .02),
                             ("budget_exhausted", .10), ("invalid_output", .10)]:
            coefficient = float(spec.get(key, default))
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError("Penalty coefficients must be finite and nonnegative")
            components[key] = coefficient * signals[key]
        cap = float(spec.get("max_penalty", .30))
        if not 0 <= cap < 1:
            raise ValueError("Penalty cap must preserve success over every failure")
        penalty = min(cap, sum(components.values()))
    else:
        penalty = 0.0
    if mode == 'business_progress':
        if row.get('split') != 'train':
            raise ValueError('Business-progress shaping is training-only')
        progress = float(row['business_progress']['score'])
        if not math.isfinite(progress) or not 0 <= progress <= .5 or cap >= .5:
            raise ValueError('Progress bonus and penalty cap must preserve complete success dominance')
        return {'mode': mode, 'official': official, 'signals': signals,
                'penalties': components, 'penalty': penalty, 'business_progress': progress,
                'total': official + progress - penalty}
    return {"mode": mode, "official": official, "signals": signals,
            "penalties": components, "penalty": penalty, "total": official - penalty}


def group_is_informative(rewards: list[float], successes: list[bool], spec: dict,
                         progresses: list[float] | None = None) -> bool:
    if len(rewards) != len(successes) or len(rewards) < 2:
        raise ValueError("A reward group needs aligned trajectories")
    if spec.get('mode') == 'business_progress' and not any(successes):
        if progresses is None or len(progresses) != len(rewards):
            raise ValueError('All-failure business-progress groups require aligned progress signals')
        return len(set(rewards)) > 1 and len(set(progresses)) > 1
    return (len(set(rewards)) > 1 and
            (spec.get("mode") != "bounded_process" or any(successes)))
