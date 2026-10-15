"""Create a local model view with a tool-call template accepting JSON strings or objects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import ROOT


def prepare_local_model(source: Path) -> Path:
    source = source.expanduser().resolve()
    config_path = source / "tokenizer_config.json"
    original = config_path.read_text(encoding="utf-8")
    config = json.loads(original)
    template = config.get("chat_template")
    if not isinstance(template, str):
        raise ValueError("Expected a local model with one tokenizer chat template")
    old = "tool_call.arguments | tojson"
    new = "tool_call.arguments if tool_call.arguments is string else (tool_call.arguments | tojson)"
    if old not in template:
        raise ValueError("Unrecognized tool template; refusing an automatic edit")
    config["chat_template"] = template.replace(old, new)
    digest = hashlib.sha256((str(source) + original + new).encode()).hexdigest()[:16]
    target = ROOT / ".cache/local-models" / source.name / digest
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name in {"tokenizer_config.json", "chat_template.jinja"}:
            continue
        link = target / entry.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(entry, target_is_directory=entry.is_dir())
        elif link.resolve() != entry.resolve():
            raise ValueError(f"Unexpected model view entry: {link}")
    (target / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Both old Transformers and newer vLLM receive the exact same template.
    (target / "chat_template.jinja").write_text(config["chat_template"], encoding="utf-8")
    (target / "retail-model-manifest.json").write_text(json.dumps({
        "source": str(source), "source_tokenizer_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "patch": "Render OpenAI JSON-string tool arguments as JSON, not double-encoded strings",
        "weight_storage": "symlinks to original files; no weights modified or copied",
    }, indent=2) + "\n", encoding="utf-8")
    return target
