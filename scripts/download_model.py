"""Download one verified HF snapshot into the project's sole model cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.multimodel import load_config  # noqa: E402


def http_resume(cfg, revision, shard_filter=None):
    """Stream official signed HF URLs to verified cache blobs, resuming by Range."""
    import httpx
    from huggingface_hub import model_info

    info = model_info(cfg["model_id"], revision=revision, files_metadata=True)
    names = [s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors")]
    if shard_filter:
        names = [name for name in names if f"-{shard_filter:05d}-of-" in name]
        if len(names) != 1:
            raise RuntimeError(f"Expected one shard numbered {shard_filter}, found {names}")
    blobs = ROOT / "hf-cache" / "hub" / ("models--" + cfg["model_id"].replace("/", "--")) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=30)) as client:
        for name in names:
            url = f"https://huggingface.co/{cfg['model_id']}/resolve/{revision}/{name}"
            head = client.head(url, follow_redirects=False)
            if head.status_code not in (200, 302):
                head.raise_for_status()
            etag = (head.headers.get("x-linked-etag") or "").strip('"')
            size = int(head.headers.get("x-linked-size") or 0)
            if len(etag) != 64 or not size:
                raise RuntimeError(f"Missing SHA-256/size metadata for {name}")
            complete = blobs / etag
            if complete.is_file() and complete.stat().st_size == size:
                print(f"Cached {name}", flush=True)
                continue
            parts = list(blobs.glob(f"{etag}.*.incomplete"))
            partial = max(parts, key=lambda p: p.stat().st_size) if parts else blobs / f"{etag}.http.incomplete"
            for attempt in range(30):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == size:
                    break
                if offset > size:
                    raise RuntimeError(f"Partial file longer than expected: {partial}")
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                print(f"{name}: {offset}/{size} bytes, attempt {attempt + 1}", flush=True)
                try:
                    with client.stream("GET", url, headers=headers) as response:
                        response.raise_for_status()
                        if offset and (response.status_code != 206 or not
                                       response.headers.get("content-range", "").startswith(f"bytes {offset}-")):
                            raise RuntimeError("CDN did not honor resume range")
                        written = offset
                        milestone = (written // (256 * 1024**2) + 1) * 256 * 1024**2
                        with partial.open("ab") as target:
                            for chunk in response.iter_bytes(chunk_size=1024**2):
                                target.write(chunk)
                                written += len(chunk)
                                if written >= milestone:
                                    print(f"{name}: {written}/{size} bytes", flush=True)
                                    milestone += 256 * 1024**2
                    if partial.stat().st_size == size:
                        break
                except (httpx.HTTPError, OSError) as exc:
                    print(f"Retrying after {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(min(5 * (attempt + 1), 30))
            if not partial.exists() or partial.stat().st_size != size:
                raise RuntimeError(f"Incomplete weight shard: {name}")
            digest = hashlib.sha256()
            with partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != etag:
                raise RuntimeError(f"SHA-256 mismatch for {name}; keeping partial for investigation")
            partial.replace(complete)
            print(f"Verified {name}: {size} bytes", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-9b")
    ap.add_argument("--http-resume", action="store_true", help="Stream verified shards by HTTP Range after a stalled Xet transfer")
    ap.add_argument("--shard", type=int, choices=(1, 2, 3, 4), help="Download one weight shard; run full command afterwards")
    args = ap.parse_args()
    cfg = load_config(args.model)
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    # Windows Xet path buffered large shards in RAM on this host; HTTP fallback
    # writes partial files to disk and can resume after an interruption.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import model_info, snapshot_download
    info = model_info(cfg["model_id"], revision=cfg["revision"])
    if info.sha != cfg["revision"]:
        raise RuntimeError("Model revision changed")
    license_name = (info.card_data or {}).get("license") if isinstance(info.card_data, dict) else getattr(info.card_data, "license", None)
    if info.id != cfg["model_id"] or license_name != "apache-2.0":
        raise RuntimeError(f"Unexpected model owner or license: {info.id} / {license_name}")
    if args.http_resume:
        http_resume(cfg, info.sha, args.shard)
    if args.shard:
        return
    snapshot = snapshot_download(cfg["model_id"], revision=info.sha,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model", "*.tiktoken", "LICENSE", "README.md"],
        max_workers=1)
    print(json.dumps({"model": cfg["model_id"], "revision": info.sha,
                      "snapshot": snapshot}, indent=2))


if __name__ == "__main__":
    main()
