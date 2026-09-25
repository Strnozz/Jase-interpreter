"""Record a real CUDA operation and host capacity for a training run."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import psutil
    import torch

    gpu = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if gpu["available"]:
        properties = torch.cuda.get_device_properties(0)
        gpu.update(name=properties.name, total_vram_bytes=properties.total_memory,
                   bf16=torch.cuda.is_bf16_supported(), compute_capability=torch.cuda.get_device_capability(0))
        left = torch.randn((1024, 1024), device="cuda")
        result = left @ left
        torch.cuda.synchronize()
        gpu["matmul_sample"] = float(result[0, 0])
        gpu["matmul_passed"] = bool(torch.isfinite(result).all())
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,memory.free",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        smi = None
    disk = shutil.disk_usage(ROOT)
    report = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "os": platform.platform(), "python": sys.version.split()[0],
              "pip": subprocess.check_output([sys.executable, "-m", "pip", "--version"], text=True).strip(),
              "torch": torch.__version__, "gpu": gpu, "nvidia_smi": smi,
              "ram_bytes": psutil.virtual_memory().total,
              "disk_free_bytes": disk.free, "disk_total_bytes": disk.total}
    (ROOT / "docs" / "MULTIMODEL_HARDWARE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if gpu.get("matmul_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
