"""Run the official orchestrator with an injected OpenAI/ART completion function.

Imports of tau2 are intentionally lazy: TAU2_DATA_DIR must be configured first.
Ground-truth actions remain inside the official task/evaluator, never in prompts.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time
from typing import Callable

from .config import configure_tau, simulator_kwargs
from .api import complete_chat
from .data import REVISION


class ModelOutputError(ValueError):
    """An invalid generated tool call, distinct from service/network failures."""


class ModelContextError(ValueError):
    """An episode exceeded the configured model context budget."""


def clean_messages(messages: list[dict]) -> list[dict]:
    result = deepcopy(messages)
    for message in result:
        if not message.get("tool_calls"):
            message.pop("tool_calls", None)
        for call in message.get("tool_calls", []):
            call.pop("name", None)  # tau's LiteLLM extension is not an OpenAI field
    return result


def make_agent(environment, completion: Callable, model: str, generation: dict):
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
    from tau2.utils.llm_utils import to_litellm_messages

    class RecordedAgent(LLMAgent):
        def __init__(self):
            super().__init__(tools=environment.get_tools(), domain_policy=environment.get_policy(), llm=model)
            self.history = []
            self.art_history = []
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
            self.tools_schema = [tool.openai_schema for tool in self.tools]
            self.truncated_generations = 0
            self.reasoning_turns = {}

        def _generate_next_message(self, message, state):
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif message is not None:
                state.messages.append(message)
            messages = clean_messages(to_litellm_messages(state.system_messages + state.messages))
            # Only new observations are added; previous choices retain ART logprobs/token IDs.
            if messages[:len(self.history)] != self.history:
                raise ValueError("tau-bench changed previous model-visible messages")
            self.art_history.extend(messages[len(self.history):])
            request_messages = deepcopy(messages)
            if generation.get("extra_body", {}).get("preserve_thinking"):
                for index, reasoning in self.reasoning_turns.items():
                    request_messages[index]["reasoning_content"] = reasoning
            from openai import BadRequestError
            try:
                response = complete_chat(completion, model=model, messages=request_messages, tools=self.tools_schema,
                                         tool_choice="auto", parallel_tool_calls=False, **generation)
            except BadRequestError as exc:
                # A long interaction exhausting its context is an episode
                # failure; other bad requests still need infrastructure repair.
                if "maximum context length" in str(exc).lower() or getattr(exc, "code", None) == "context_length_exceeded":
                    raise ModelContextError("Episode exhausted its model context budget") from exc
                raise
            choice = response.choices[0]
            reasoning = getattr(choice.message, "reasoning_content", None)
            if reasoning:
                self.reasoning_turns[len(messages)] = reasoning
            self.art_history.append(choice)
            self.history = messages + [choice.message.model_dump(exclude_none=True)]
            if choice.finish_reason == "length":
                self.truncated_generations += 1
            usage = response.usage
            for key in self.usage:
                self.usage[key] += int(getattr(usage, key, 0) or 0)
            tool_calls = []
            for call in choice.message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments)
                except (ValueError, TypeError) as exc:
                    raise ModelOutputError("Invalid JSON tool arguments") from exc
                if not isinstance(arguments, dict):
                    raise ModelOutputError("Model tool arguments are not a JSON object")
                tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
            answer = AssistantMessage(role="assistant", content=choice.message.content,
                                      tool_calls=tool_calls or None,
                                      usage={"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                                             "output_tokens": getattr(usage, "completion_tokens", 0) or 0})
            self.history = clean_messages(to_litellm_messages(state.system_messages + state.messages + [answer]))
            if not answer.has_content() and not answer.is_tool_call():
                raise ModelOutputError("Model returned an empty assistant message")
            return answer

    return RecordedAgent()


def run_episode(config: dict, task_id: str, split: str, completion: Callable, model: str,
                *, source: str, seed: int, tasks_sha256: str, generation: dict,
                user_override=None) -> tuple[dict, list]:
    configure_tau(config)
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
    from tau2.user.user_simulator import UserSimulator
    from tau2.data_model.message import ToolMessage

    task = next(t for t in get_tasks() if t.id == task_id)
    judge_outputs = []
    needs_nl = task.evaluation_criteria and "NL_ASSERTION" in [getattr(x, "value", x) for x in task.evaluation_criteria.reward_basis]
    if needs_nl:
        # Pinned upstream's judge is module-configured. Keep it fixed for an entire run.
        import tau2.evaluator.evaluator_nl_assertions as nl
        spec = config["judge"]
        nl.DEFAULT_LLM_NL_ASSERTIONS = spec["model"]
        nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = simulator_kwargs(spec)
        from .judging import configure_judge
        configure_judge(nl, task.evaluation_criteria.nl_assertions, judge_outputs)
    environment = get_environment()
    agent = make_agent(environment, completion, model, generation)
    if user_override is None:
        spec = config["user"]
        if config.get("user_grounding") == "visible-identifiers-v1":
            from .grounding import make_grounded_user
            user = make_grounded_user(spec, str(task.user_scenario), simulator_kwargs(spec))
        else:
            user = UserSimulator(tools=None, instructions=str(task.user_scenario), llm=spec["model"],
                                 llm_args=simulator_kwargs(spec))
    else:
        user = user_override
    orchestrator = Orchestrator(domain="retail", agent=agent, user=user, environment=environment,
                                task=task, max_steps=config["max_steps"], max_errors=config["max_errors"],
                                seed=seed, timeout=600)
    started = time.monotonic()
    # Exceptions propagate: failed infrastructure must never become RL reward=0.
    try:
        simulation = orchestrator.run()
    except (ModelOutputError, ModelContextError) as exc:
        context_exhausted = isinstance(exc, ModelContextError)
        partial = orchestrator.trajectory
        errors = sum(isinstance(m, ToolMessage) and m.requestor == "assistant" and bool(m.error) for m in partial)
        return {"schema_version": 2, "domain": "retail", "task_id": task_id, "split": split,
                "source": source, "seed": seed, "model": model, "tau_revision": REVISION,
                "tasks_sha256": tasks_sha256, "success": False, "official_reward": 0.0,
                "scoring_note": "Model output/context budget terminated the episode; assigned failure, not evaluated as a completed task",
                "termination": "context_length_exceeded" if context_exhausted else "invalid_model_output",
                "tool_calls": sum(len(m.get("tool_calls", [])) for m in agent.history),
                "tool_errors": errors, "model_output_errors": int(not context_exhausted),
                "context_overflow": int(context_exhausted),
                "truncated_generations": agent.truncated_generations, "usage": agent.usage,
                "user_grounding_events": getattr(user, "grounding_events", []),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                "messages": agent.history, "tools": agent.tools_schema,
                "simulation": {"partial_messages": [m.model_dump(mode="json") for m in partial]},
                "policy_compliance": "not_exhaustively_checked"}, agent.art_history
    simulation.policy = environment.get_policy()
    from pydantic import ValidationError
    evaluation_retries = []
    for attempt in range(3):
        try:
            reward_info = evaluate_simulation(simulation=simulation, task=task, evaluation_type=EvaluationType.ALL,
                                              solo_mode=False, domain="retail")
            # Upstream assigns reward=0 without invoking any judge on premature
            # termination (max_steps/tool-error limit). That is a valid failure.
            expected = task.evaluation_criteria.nl_assertions if needs_nl and reward_info.reward_basis else []
            if expected and (len(reward_info.nl_assertions or []) != len(expected)
                             or {check.nl_assertion for check in reward_info.nl_assertions} != set(expected)):
                print("judge_coverage_mismatch=" + json.dumps({"task_id": task_id, "expected": expected,
                      "observed": [check.nl_assertion for check in reward_info.nl_assertions or []]}), flush=True)
                raise ValueError("Judge did not return exactly the required assertions")
            break
        except (json.JSONDecodeError, KeyError, ValidationError, ValueError) as exc:
            evaluation_retries.append(type(exc).__name__)
            if attempt == 2:
                raise
    simulation.reward_info = reward_info
    errors = [m for m in simulation.messages if isinstance(m, ToolMessage) and m.requestor == "assistant" and m.error]
    calls = sum(len(m.get("tool_calls", [])) for m in agent.history)
    row = {"schema_version": 2, "domain": "retail", "task_id": task_id, "split": split, "source": source,
           "seed": seed, "model": model, "tau_revision": REVISION, "tasks_sha256": tasks_sha256,
           "success": reward_info.reward == 1.0, "official_reward": reward_info.reward,
           "termination": simulation.termination_reason.value, "tool_calls": calls, "tool_errors": len(errors),
           "truncated_generations": agent.truncated_generations,
           "user_grounding_events": getattr(user, "grounding_events", []),
           "evaluation_retries": evaluation_retries,
           "judge_outputs": judge_outputs,
           "elapsed_ms": round((time.monotonic() - started) * 1000, 2), "usage": agent.usage,
           "messages": agent.history, "tools": agent.tools_schema,
           "simulation": simulation.model_dump(mode="json"),
           "policy_compliance": "not_exhaustively_checked"}
    return row, agent.art_history


def training_reward(row: dict, error_penalty: float = 0.0) -> float:
    """Default exactly matches official reward. Process shaping is explicit and separate."""
    if row.get("error"):
        raise ValueError("Infrastructure failures cannot be training samples")
    return float(row["official_reward"]) - error_penalty * row["tool_errors"]
