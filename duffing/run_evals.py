"""Evaluate every trained run (fresh, overwriting debug evals), keeping OUR train.py + evaluate_run.py
processes <= P.  usage: python run_evals.py [P] [PATTERN]"""
import subprocess
import sys
import time

import psutil

from common import ROOT
(ROOT / "logs").mkdir(exist_ok=True)

P = int(sys.argv[1]) if len(sys.argv) > 1 else 6
pat = sys.argv[2] if len(sys.argv) > 2 else "*"
ME = str(ROOT).lower()


def ours():
    n = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            c = p.info["cmdline"] or []
            if p.info["name"] == "python.exe" and ("train.py" in c or "evaluate_run.py" in c) and p.cwd().lower() == ME:
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


tags = sorted(d.name for d in (ROOT / "runs").glob(pat) if (d / "model.pt").exists())
# slow MMN probe evaluations first
tags.sort(key=lambda t: (0 if ("mmn" in t and t.startswith("A0_")) else 1, t))
procs = []
for t in tags:
    while ours() >= P:
        time.sleep(5)
    f = open(ROOT / "logs" / f"eval_{t}.log", "w")
    procs.append((t, subprocess.Popen([sys.executable, "evaluate_run.py", t], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)))
    time.sleep(3)
for t, p in procs:
    p.wait()
    if p.returncode:
        print("FAILED", t, flush=True)
print("EVAL DONE", len(procs), time.strftime("%H:%M:%S"), flush=True)
