import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from jinja2 import Environment

from mini_customer_agent.retail.local_model import prepare_local_model
from mini_customer_agent.retail.training import check_sft_lengths


class LocalModelTests(unittest.TestCase):
    def test_template_renders_json_strings_and_objects_without_changing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model"
            source.mkdir()
            original = json.dumps({"chat_template": "{{ tool_call.arguments | tojson }}"})
            (source / "tokenizer_config.json").write_text(original)
            (source / "model.safetensors").write_bytes(b"test-only-weight-placeholder")
            with patch("mini_customer_agent.retail.local_model.ROOT", root):
                target = prepare_local_model(source)
                self.assertEqual(prepare_local_model(source), target)
            self.assertEqual((source / "tokenizer_config.json").read_text(), original)
            self.assertTrue((target / "model.safetensors").is_symlink())
            self.assertEqual((target / "model.safetensors").resolve(), source / "model.safetensors")
            template = Environment().from_string((target / "chat_template.jinja").read_text())
            for arguments in ('{"order_id":"A"}', {"order_id": "A"}):
                rendered = template.render(tool_call={"arguments": arguments})
                self.assertEqual(json.loads(rendered), {"order_id": "A"})

    def test_length_guard_counts_input_tokens_not_encoding_dictionary_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            data.write_text(json.dumps({"messages": [{"role": "assistant", "content": "test"}], "tools": []}))
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: {
                "input_ids": [1] * 5000, "attention_mask": [1] * 5000})
            module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
            with patch.dict("sys.modules", {"transformers": module}):
                with self.assertRaisesRegex(ValueError, "5000"):
                    check_sft_lengths(data, {"base_model": "test", "max_seq_length": 4096})
