"""Supervise the checkpoint evaluation with explicitly approved temporary sleep prevention."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from scripts.evaluate_checkpoint_robustness import OUTPUT, ROOT


def main() -> None:
    """Record one exclusive attempt and release the Windows request on every exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "background_status.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write("{}\n")
    status = {
        "state": "starting",
        "supervisor_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "keep_awake_requested": args.keep_awake,
        "keep_awake_active": False,
    }

    def save() -> None:
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if args.keep_awake:
            if os.name != "nt" or not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                raise RuntimeError("Temporary sleep prevention unavailable")
            status["keep_awake_active"] = True
        with (OUTPUT / "stdout.log").open("x", encoding="utf-8") as stdout:
            with (OUTPUT / "stderr.log").open("x", encoding="utf-8") as stderr:
                child = subprocess.Popen(  # noqa: S603 - fixed local module and interpreter
                    [sys.executable, "-u", "-m", "scripts.evaluate_checkpoint_robustness"],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                status.update(state="running", child_pid=child.pid)
                save()
                status["exit_code"] = child.wait()
        status["state"] = "completed" if status["exit_code"] == 0 else "failed"
    except Exception as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        if status["keep_awake_active"]:
            status["keep_awake_release_succeeded"] = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000))
            status["keep_awake_active"] = False
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
