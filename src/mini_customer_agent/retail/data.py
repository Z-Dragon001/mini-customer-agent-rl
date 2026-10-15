"""Fetch a pinned, hash-verified upstream source/data subset without large voice assets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
REVISION = "5bfa7e37b36656b37dc6d022156be6563c1007f3"
REPOSITORY = "sierra-research/tau2-bench"
UPSTREAM = ROOT / "third_party" / "tau2-bench"
DATA = ROOT / "data" / "tau2"
RETAIL_FILES = {"db.json", "policy.md", "tasks.json", "split_tasks.json"}


def fetch(url: str, *, direct: bool = False) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    request = urllib.request.Request(url, headers={"User-Agent": "ServiceAgent-RL-data-setup"})
    for attempt in range(3):
        try:
            with opener.open(request, timeout=45) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Unreachable")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(*, direct: bool = False) -> dict:
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1", direct=direct))
    if tree.get("truncated"):
        raise ValueError("Upstream tree is truncated")
    entries = []
    for item in tree["tree"]:
        path = item["path"]
        if item["type"] != "blob":
            continue
        if path.startswith("src/tau2/") or (path.startswith("data/tau2/user_simulator/") and path.endswith(".md")) or path in {"pyproject.toml", "README.md", "LICENSE"} or (
            path.startswith("data/tau2/domains/retail/") and Path(path).name in RETAIL_FILES
        ):
            entries.append(item)

    def download(item: dict) -> dict:
        target = UPSTREAM / item["path"]
        def matches(content: bytes) -> bool:
            header = f"blob {len(content)}\0".encode()
            return hashlib.sha1(header + content).hexdigest() == item["sha"]
        if not target.exists() or not matches(target.read_bytes()):
            content = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{item['path']}", direct=direct)
            if not matches(content):
                raise ValueError(f"Upstream hash mismatch: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return {"path": item["path"], "sha256": sha256(target), "git_blob": item["sha"]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(download, entries))
    manifest = {"repository": f"https://github.com/{REPOSITORY}", "revision": REVISION,
                "license": "MIT; see third_party/tau2-bench/LICENSE", "files": files}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "upstream-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify() -> dict:
    manifest = json.loads((DATA / "upstream-manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected tau-bench version; run retail-prepare")
    for item in manifest["files"]:
        if sha256(UPSTREAM / item["path"]) != item["sha256"]:
            raise ValueError(f"Modified upstream file: {item['path']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", action="store_true", help="Bypass environment proxies for this download only")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else prepare(direct=args.direct)
    print(json.dumps({"revision": result["revision"], "files": len(result["files"]), "source": str(UPSTREAM)}, indent=2))


if __name__ == "__main__":
    main()
