from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
from copy import deepcopy

from .data import ROOT, REVISION


def load_config(path: str | Path) -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["domain"] != "retail":
        raise ValueError("This integration pins the retail domain")
    # Keep secrets out of persisted experiment configurations.
    def check(value):
        if isinstance(value, dict):
            if "api_key" in value:
                raise ValueError("Use api_key_env, never literal API keys in config")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(config)
    for role in ("agent", "teacher", "user", "judge"):
        generation_kwargs(config[role], allow_stream=role in {"agent", "teacher"})
    return config


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_tau(config: dict) -> None:
    data_dir = project_path(config["data_dir"]).resolve()
    if data_dir != (ROOT / "third_party/tau2-bench/data").resolve():
        raise ValueError("Use the pinned data directory so verification and execution use the same tasks")
    if not (data_dir / "tau2/domains/retail/tasks.json").exists():
        raise FileNotFoundError("Run retail-prepare before using tau-bench")
    # tau2 resolves DATA_DIR at import time. Never import tau2 before this call.
    os.environ["TAU2_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def endpoint_kwargs(spec: dict) -> dict:
    model = spec["model"]
    if "REPLACE_" in model:
        raise ValueError("Configure a real model in configs/retail.json")
    key = os.getenv(spec["api_key_env"]) or os.getenv(spec.get("api_key_env_fallback", ""))
    if not key:
        names = " / ".join(filter(None, [spec["api_key_env"], spec.get("api_key_env_fallback")]))
        raise ValueError(f"Missing environment variable: {names}")
    base_url = (os.getenv(spec.get("base_url_env", ""))
                or os.getenv(spec.get("base_url_env_fallback", "")) or spec.get("base_url"))
    if not base_url:
        raise ValueError("Configure base_url or base_url_env for each endpoint")
    return {"api_key": key, "base_url": base_url, "timeout": 60.0, "max_retries": 2}


def generation_kwargs(spec: dict, *, allow_stream: bool = True) -> dict:
    extra = spec.get("extra_body", {})
    if not isinstance(extra, dict):
        raise ValueError("extra_body must be a JSON object")
    # These fields are controlled by the harness, not provider extensions.
    reserved = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
                "stream_options", "temperature", "max_tokens", "n", "api_key"}
    if reserved & extra.keys():
        raise ValueError("extra_body cannot override model, messages, tools or generation settings")
    for name in ("enable_thinking", "preserve_thinking"):
        if name in extra and not isinstance(extra[name], bool):
            raise ValueError(f"{name} must be a JSON boolean")
    stream = spec.get("stream", False)
    if not isinstance(stream, bool) or (stream and not allow_stream):
        raise ValueError("stream must be boolean; user/judge currently require stream=false")
    result = {"temperature": spec["temperature"], "max_tokens": spec["max_tokens"]}
    if extra:
        result["extra_body"] = deepcopy(extra)
    if allow_stream:
        result["stream"] = stream
        if stream:
            result["stream_options"] = {"include_usage": True}
    return result


def simulator_kwargs(spec: dict) -> dict:
    connection = endpoint_kwargs(spec)
    return {"api_key": connection["api_key"], "api_base": connection["base_url"],
            "timeout": connection["timeout"], "num_retries": connection["max_retries"],
            **generation_kwargs(spec, allow_stream=False)}


def provenance(config: dict) -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    versions = {}
    for package in ("openpipe-art", "tau2", "openai", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"config": config, "code_revision": revision.stdout.strip(), "dirty": bool(dirty.stdout.strip()),
            "tau_revision": REVISION, "packages": versions,
            "retail_source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted((ROOT/'src/mini_customer_agent/retail').glob('*.py'))}}
