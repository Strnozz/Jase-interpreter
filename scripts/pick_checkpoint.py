#!/usr/bin/env python3
"""Estrae un checkpoint intermedio in una cartella adapter valutabile.

MLX salva i checkpoint come `0002000_adapters.safetensors` dentro
l'adapter_path, ma `mlx_lm.load()` carica solo `adapters.safetensors`.
Questo script crea una cartella per il checkpoint scelto, così ogni
iterazione può essere valutata sul gold invece di fidarsi della
validation loss — che nel v1 aveva già mentito una volta (iter400 aveva
loss peggiore di iter300 e generalizzava meglio).

    python scripts/pick_checkpoint.py adapters/jase-v3 2000
    → adapters/jase-v3-iter2000/
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src = ROOT / sys.argv[1]
    iters = sys.argv[2:]
    cfg = src / "adapter_config.json"
    if not cfg.exists():
        print(f"manca {cfg}", file=sys.stderr)
        return 1
    for it in iters:
        ckpt = src / f"{int(it):07d}_adapters.safetensors"
        if not ckpt.exists():
            print(f"  salto iter {it}: {ckpt.name} non c'è")
            continue
        dst = ROOT / f"{sys.argv[1]}-iter{int(it)}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, dst / "adapter_config.json")
        shutil.copy2(ckpt, dst / "adapters.safetensors")
        print(f"  {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
