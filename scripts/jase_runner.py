#!/usr/bin/env python3
"""Job runner locale per il ciclo train/eval di Jase.

PERCHÉ ESISTE
    L'agente che sviluppa Jase lavora su una VM Linux con la cartella del
    progetto montata: può leggere e scrivere i file, ma non può eseguire MLX,
    che gira solo su macOS nativo. Questo runner colma il vuoto: lo lanci una
    volta in un Terminale e resta in ascolto; l'agente deposita i job nella
    coda, il runner li esegue QUI e scrive i risultati nel repo, dove l'agente
    li rilegge.

COME SI USA
    cd ~/Desktop/jase-interpreter-starter
    .venv/bin/python scripts/jase_runner.py

    Ctrl-C per fermarlo. Ogni comando viene stampato prima di partire.

COSA PUÒ ESEGUIRE
    Solo gli eseguibili in ALLOWED (python del venv e gli entrypoint mlx_lm),
    solo con working directory dentro il repo, solo come lista di argomenti
    (nessuna shell, quindi nessuna pipe, nessun redirect, nessun `rm`).
    Un job che chiede altro viene rifiutato e registrato come tale.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs" / "queue"
DONE = ROOT / "runs" / "done"
LOGS = ROOT / "runs" / "logs"
RUNNING = ROOT / "runs" / "running.json"
HEARTBEAT = ROOT / "runs" / "heartbeat.json"

#: eseguibili ammessi, relativi alla radice del repo
ALLOWED = {
    ".venv/bin/python",
    ".venv/bin/python3",
    ".venv/bin/mlx_lm.lora",
    ".venv/bin/mlx_lm.generate",
    ".venv/bin/mlx_lm.fuse",
    ".venv/bin/mlx_lm.server",
}

DEFAULT_TIMEOUT = 60 * 90  # 90 minuti: un training LoRA lungo ci sta dentro


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tail(path: Path, n: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def validate(job: dict) -> str | None:
    """Ritorna il motivo del rifiuto, oppure None se il job è ammissibile."""
    cmd = job.get("cmd")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
        return "cmd deve essere una lista di stringhe"
    exe = cmd[0]
    if exe not in ALLOWED:
        return f"eseguibile non consentito: {exe!r} (ammessi: {sorted(ALLOWED)})"
    if not (ROOT / exe).exists() and not (ROOT / exe).is_symlink():
        return f"eseguibile non trovato nel repo: {exe}"
    for arg in cmd[1:]:
        if arg.startswith("/") and not arg.startswith(str(ROOT)):
            return f"percorso assoluto fuori dal repo: {arg!r}"
        if ".." in Path(arg).parts:
            return f"percorso con '..': {arg!r}"
    return None


def run_job(path: Path, verbose: bool = True) -> dict:
    job = json.loads(path.read_text(encoding="utf-8"))
    jid = job.get("id") or path.stem
    log_path = LOGS / f"{jid}.log"
    started = now()

    reason = validate(job)
    if reason:
        print(f"  RIFIUTATO {jid}: {reason}")
        return {"id": jid, "status": "refused", "reason": reason,
                "started": started, "finished": now()}

    cmd = [str(ROOT / job["cmd"][0])] + job["cmd"][1:]
    timeout = int(job.get("timeout", DEFAULT_TIMEOUT))
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"  ESEGUO {jid}")
    print(f"    {shlex.join(job['cmd'])}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {started}\n# {shlex.join(job['cmd'])}\n\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            code, status = proc.returncode, ("ok" if proc.returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            code, status = -1, "timeout"
            log.write(f"\n\n# TIMEOUT dopo {timeout}s\n")
        except Exception as exc:  # noqa: BLE001
            code, status = -1, "error"
            log.write(f"\n\n# ERRORE RUNNER: {exc}\n")

    elapsed = round(time.time() - t0, 1)
    print(f"    → {status} ({code}) in {elapsed}s")
    return {"id": jid, "status": status, "exit_code": code,
            "seconds": elapsed, "started": started, "finished": now(),
            "cmd": job["cmd"], "log": str(log_path.relative_to(ROOT)),
            "tail": tail(log_path, int(job.get("tail_lines", 160)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true",
                    help="svuota la coda e termina, invece di restare in ascolto")
    args = ap.parse_args()

    for d in (QUEUE, DONE, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Jase runner attivo su {ROOT}")
    print(f"  coda:      {QUEUE.relative_to(ROOT)}")
    print(f"  risultati: {DONE.relative_to(ROOT)}")
    print("  Ctrl-C per fermare.\n")

    stop = False

    def _sigint(_s, _f):
        nonlocal stop
        stop = True
        print("\nChiusura richiesta, esco dopo il job corrente.")

    signal.signal(signal.SIGINT, _sigint)

    while not stop:
        jobs = sorted(QUEUE.glob("*.job"))
        if not jobs:
            HEARTBEAT.write_text(json.dumps({"alive": now(), "pending": 0}),
                                 encoding="utf-8")
            if args.once:
                break
            time.sleep(args.interval)
            continue

        for jp in jobs:
            if stop:
                break
            RUNNING.write_text(json.dumps({"id": jp.stem, "since": now()}),
                               encoding="utf-8")
            try:
                result = run_job(jp)
            except Exception as exc:  # noqa: BLE001
                result = {"id": jp.stem, "status": "error", "reason": str(exc),
                          "finished": now()}
            (DONE / f"{result['id']}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            jp.unlink(missing_ok=True)
            RUNNING.unlink(missing_ok=True)

    HEARTBEAT.write_text(json.dumps({"alive": now(), "stopped": True}),
                         encoding="utf-8")
    print("Runner fermato.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
