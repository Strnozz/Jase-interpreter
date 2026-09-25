#!/usr/bin/env python3
"""Media dei pesi di più checkpoint LoRA ("model soup").

PERCHÉ
    Fra un checkpoint e l'altro della STESSA corsa i casi mandatory oscillano
    di tre o quattro casi (v14: 0.788 a 16 000, 0.859 a 17 000). Non è
    apprendimento, è rumore: l'ottimizzatore rimbalza attorno a un minimo con
    il learning rate costante. Mediare i pesi di più checkpoint è il modo più
    economico di togliere quel rumore senza riallenare.

    Non è un trucco sul benchmark: produce un adattatore vero, che si spedisce
    come qualsiasi altro, e il gold lo giudica con lo stesso metro.

COME
    Media aritmetica tensore per tensore. I checkpoint devono venire dalla
    stessa corsa (stessa forma dei tensori, stessa configurazione).

    python scripts/soup_adapters.py adapters/jase-v14-soup \
        adapters/jase-v14/0012000_adapters.safetensors \
        adapters/jase-v14/0014000_adapters.safetensors \
        adapters/jase-v14/adapters.safetensors
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    out_dir = Path(sys.argv[1])
    parts = [Path(p) for p in sys.argv[2:]]
    for p in parts:
        if not p.exists():
            print(f"manca: {p}")
            return 1

    import mlx.core as mx

    acc: dict[str, "mx.array"] = {}
    for i, p in enumerate(parts):
        w = mx.load(str(p))
        if i == 0:
            acc = {k: v.astype(mx.float32) for k, v in w.items()}
        else:
            if set(w) != set(acc):
                print(f"chiavi diverse in {p.name}: non sono della stessa corsa")
                return 1
            for k in acc:
                if w[k].shape != acc[k].shape:
                    print(f"forma diversa per {k} in {p.name}")
                    return 1
                acc[k] = acc[k] + w[k].astype(mx.float32)
    n = len(parts)
    merged = {k: (v / n).astype(mx.float16) for k, v in acc.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), merged)
    # la configurazione si eredita dal primo checkpoint: la soup non cambia
    # rango, scala o strati, solo i valori
    src_cfg = parts[0].parent / "adapter_config.json"
    if src_cfg.exists():
        shutil.copy(src_cfg, out_dir / "adapter_config.json")
    print(f"media di {n} checkpoint → {out_dir}/adapters.safetensors")
    print("  " + "\n  ".join(p.name for p in parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
