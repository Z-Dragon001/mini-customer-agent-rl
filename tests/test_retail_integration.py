"""Offline integration checks, never demonstrations or model performance evidence."""
import importlib.util
import asyncio
import json
import os
import unittest
from unittest.mock import patch

from mini_customer_agent.retail.config import load_config, configure_tau
from mini_customer_agent.retail.data import ROOT, DATA, verify


@unittest.skipUnless(importlib.util.find_spec("tau2") and (DATA / "upstream-manifest.json").exists(),
                     "Install retail dependencies and run retail-prepare")
class OfficialRetailIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/retail.example.json")
        configure_tau(cls.config)
        from loguru import logger
        logger.disable("tau2")

    def test_official_tools_state_evaluator_and_art_trajectory(self):
        from openai.types.chat import ChatCompletion
        import art
        from tau2.domains.retail.environment import get_tasks
        from tau2.user.user_simulator import UserSimulator
        from tau2.data_model.message import UserMessage
        from mini_customer_agent.retail.rollout import run_episode
        from mini_customer_agent.retail.dataset import load_splits
        verify()
        task = next(t for t in get_tasks() if t.id == "33")
        actions = iter(task.evaluation_criteria.actions)
        requests = []

        class OfflineUser(UserSimulator):
            def _generate_next_message(self, message, state):
                content = "###STOP###" if message and message.content == "Completed." else "Please update my address. Yes, I confirm."
                return UserMessage(role="user", content=content)

        def completion(**request):
            requests.append(request)
            action = next(actions, None)
            message = {"role": "assistant", "content": "Completed."}
            if action:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": action.action_id, "type": "function",
                    "function": {"name": action.name, "arguments": json.dumps(action.arguments)}}]}
            message["reasoning_content"] = "Synthetic test reasoning; not an SFT target."
            return ChatCompletion(id="offline", object="chat.completion", created=0, model="offline-test",
                                  choices=[{"index": 0, "finish_reason": "tool_calls" if action else "stop", "message": message}])

        with patch("httpx.Client.send", side_effect=AssertionError("Offline test attempted network access")):
            row, history = run_episode(self.config, task.id, "test", completion, "offline-test",
                                       source="test_oracle", seed=42,
                                       tasks_sha256=load_splits(DATA / "splits.json")["tasks_sha256"],
                                       generation={"extra_body": {"enable_thinking": True, "preserve_thinking": True}},
                                       user_override=OfflineUser(instructions="Offline test", llm="offline"))
        self.assertTrue(row["success"])
        self.assertEqual(row["tool_calls"], 6)
        self.assertEqual(row["tool_errors"], 0)
        self.assertEqual(row["termination"], "user_stop")
        trajectory = art.Trajectory(messages_and_choices=history, tools=row["tools"], reward=row["official_reward"])
        self.assertEqual(trajectory.reward, 1)
        self.assertEqual(len(trajectory.messages()), len(row["messages"]))
        self.assertEqual(len(requests), 7)
        self.assertNotIn("resolution steps", requests[0]["messages"][0]["content"])
        self.assertTrue(any(m.get("reasoning_content") for m in requests[1]["messages"]))
        self.assertFalse(any("reasoning_content" in m for m in row["messages"]))

    def test_art_configuration_matches_pinned_api(self):
        from art import TrainableModel, dev
        from mini_customer_agent.retail.training import model_config, require_single_gpu
        config = model_config(self.config["art"])
        self.assertLessEqual(config.keys(), dev.InternalModelConfig.__annotations__.keys())
        for name, schema in [("init_args", dev.InitArgs), ("peft_args", dev.PeftArgs), ("trainer_args", dev.TrainerArgs), ("engine_args", dev.EngineArgs)]:
            self.assertLessEqual(config[name].keys(), schema.__annotations__.keys())
        model = TrainableModel(name="offline", project="test", base_model="Qwen/Qwen2.5-3B-Instruct", _internal_config=config)
        self.assertEqual(model.base_model, "Qwen/Qwen2.5-3B-Instruct")
        for value in ("", "0,1", "-1"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}), self.assertRaises(ValueError):
                require_single_gpu()

    def test_invalid_model_json_is_distinct_from_infrastructure_failure(self):
        from openai.types.chat import ChatCompletion
        from tau2.user.user_simulator import UserSimulator
        from tau2.data_model.message import UserMessage
        from mini_customer_agent.retail.rollout import run_episode
        from mini_customer_agent.retail.dataset import load_splits

        class OfflineUser(UserSimulator):
            def _generate_next_message(self, message, state):
                return UserMessage(role="user", content="Please look up my order.")

        def malformed(**request):
            return ChatCompletion(id="offline", object="chat.completion", created=0, model="offline-test",
                                  choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                                      "role": "assistant", "content": None, "tool_calls": [{
                                          "id": "bad", "type": "function", "function": {
                                              "name": "get_order_details", "arguments": "{invalid"}}]}}])

        row, _ = run_episode(self.config, "33", "test", malformed, "offline-test", source="test_oracle",
                             seed=42, tasks_sha256=load_splits(DATA / "splits.json")["tasks_sha256"],
                             generation={}, user_override=OfflineUser(instructions="Offline", llm="offline"))
        self.assertEqual(row["termination"], "invalid_model_output")
        self.assertEqual(row["official_reward"], 0)
        self.assertNotIn("error", row)
        self.assertEqual(row["messages"][-1]["tool_calls"][0]["function"]["arguments"], "{invalid")

        def empty(**request):
            return ChatCompletion(id="offline", object="chat.completion", created=0, model="offline-test",
                                  choices=[{"index": 0, "finish_reason": "stop", "message": {
                                      "role": "assistant", "content": ""}}])

        row, _ = run_episode(self.config, "33", "test", empty, "offline-test", source="test_oracle",
                             seed=42, tasks_sha256="offline", generation={},
                             user_override=OfflineUser(instructions="Offline", llm="offline"))
        self.assertEqual(row["termination"], "invalid_model_output")
        self.assertEqual(row["official_reward"], 0)
        self.assertNotIn("error", row)

    def test_context_budget_failure_is_not_a_transient_api_error(self):
        import httpx
        from openai import BadRequestError
        from tau2.user.user_simulator import UserSimulator
        from tau2.data_model.message import UserMessage
        from mini_customer_agent.retail.rollout import run_episode

        class OfflineUser(UserSimulator):
            def _generate_next_message(self, message, state):
                return UserMessage(role="user", content="Please help.")

        def run_error(message):
            def completion(**kwargs):
                response = httpx.Response(400, request=httpx.Request("POST", "http://offline/v1/chat/completions"))
                raise BadRequestError(message, response=response, body={"message": message})
            return run_episode(self.config, "33", "test", completion, "offline", source="test_oracle",
                               seed=42, tasks_sha256="offline", generation={},
                               user_override=OfflineUser(instructions="Offline", llm="offline"))

        row, _ = run_error("This model's maximum context length is 16384 tokens")
        self.assertEqual(row["termination"], "context_length_exceeded")
        self.assertEqual(row["official_reward"], 0)
        self.assertNotIn("error", row)
        with self.assertRaises(BadRequestError):
            run_error("Unsupported request option")

    def test_art_client_has_numeric_timeouts_with_installed_sdk(self):
        from art import TrainableModel
        model = TrainableModel(name="offline", project="test", base_model="offline-test")
        model.inference_base_url = "http://127.0.0.1:1/v1"
        model.inference_api_key = "test-only"
        client = model.openai_client()
        try:
            # OpenAI 3.x/httpx2 interpreted ART's httpx.Timeout as a scalar,
            # causing TypeError before even attempting a connection.
            self.assertIsInstance(client._client.timeout.connect, (int, float))
            self.assertIsInstance(client._client.timeout.read, (int, float))
        finally:
            asyncio.run(client.close())

    def test_premature_termination_needs_no_nl_judge_results(self):
        from openai.types.chat import ChatCompletion
        from tau2.user.user_simulator import UserSimulator
        from tau2.data_model.message import UserMessage
        from mini_customer_agent.retail.rollout import run_episode
        from mini_customer_agent.retail.dataset import load_splits

        class OfflineUser(UserSimulator):
            def _generate_next_message(self, message, state):
                return UserMessage(role="user", content="Please help with my order.")

        def answer(**kwargs):
            return ChatCompletion(id="offline", object="chat.completion", created=0, model="offline",
                                  choices=[{"index": 0, "finish_reason": "stop", "message": {
                                      "role": "assistant", "content": "Please tell me more."}}])

        config = {**self.config, "max_steps": 2}
        with patch("mini_customer_agent.retail.rollout.simulator_kwargs", return_value={}):
            row, _ = run_episode(config, "43", "train", answer, "offline", source="test_oracle", seed=42,
                                 tasks_sha256=load_splits(DATA / "splits.json")["tasks_sha256"], generation={},
                                 user_override=OfflineUser(instructions="Offline", llm="offline"))
        self.assertEqual(row["termination"], "max_steps")
        self.assertEqual(row["official_reward"], 0)
        self.assertEqual(row["judge_outputs"], [])
        self.assertNotIn("error", row)


if __name__ == "__main__":
    unittest.main()
