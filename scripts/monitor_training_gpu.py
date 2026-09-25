"""Sample physical GPU usage while a long local training run is active."""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--max-hours", type=float, default=6)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = run / "manifest.json"
    if not manifest.is_file():
        parser.error(f"Training manifest missing: {manifest}")
    # Keep the Windows host awake for this bounded training monitor only.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    path = run / "gpu_monitor.csv"
    fields = ["at_utc", "gpu_util_percent", "gpu_memory_used_mib", "gpu_power_w",
              "gpu_temperature_c", "system_ram_used_bytes"]
    deadline = time.monotonic() + args.max_hours * 3600
    try:
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if path.stat().st_size == 0:
                writer.writeheader()
            while time.monotonic() < deadline:
                state = json.loads(manifest.read_text(encoding="utf-8"))
                if state.get("status") in ("complete", "failed"):
                    break
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"], text=True, capture_output=True,
                    timeout=15, check=True)
                values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
                writer.writerow(dict(zip(fields, [datetime.now(timezone.utc).isoformat(),
                                                    *values, psutil.virtual_memory().used])))
                file.flush()
                time.sleep(args.interval)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
