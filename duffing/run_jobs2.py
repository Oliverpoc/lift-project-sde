"""Dynamic job launcher: keeps at most P of OUR train.py processes running (counted by working
directory = this folder, so jobs started by an earlier runner are counted too and never duplicated).
usage: python run_jobs2.py JOBFILE [P]"""
import subprocess
import sys
import time
from pathlib import Path

import psutil

from common import ROOT, run_tag
(ROOT / "logs").mkdir(exist_ok=True)

jobfile, P = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
lines = [l.split() for l in Path(jobfile).read_text().splitlines() if l.strip() and not l.startswith("#")]
ME = str(ROOT).lower()


def ours():
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if p.info["name"] == "python.exe" and p.info["cmdline"] and "train.py" in p.info["cmdline"] \
                    and p.cwd().lower() == ME:
                a = p.info["cmdline"][p.info["cmdline"].index("train.py") + 1:]
                out.append(run_tag(a[0], a[1], a[2], int(a[4]) if len(a) > 4 else 0, int(a[3])))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


procs = {}
for args in lines:
    w, ex, m, s = args[0], args[1], args[2], int(args[3]); dh = int(args[4]) if len(args) > 4 else 0
    tag = run_tag(w, ex, m, dh, s)
    if (ROOT / "runs" / tag / "model.pt").exists() or tag in ours():
        continue
    if time.strftime("%H:%M") >= "23:30" or time.strftime("%H:%M") < "06:00":
        print("cutoff reached, not starting", tag, flush=True); continue
    while len(ours()) >= P:
        for t, (pr, t0) in list(procs.items()):
            if pr.poll() is not None:
                print(f"{time.strftime('%H:%M:%S')} {t} rc={pr.returncode} {time.time()-t0:.0f}s", flush=True)
                del procs[t]
        time.sleep(20)
    f = open(ROOT / "logs" / f"{tag}.log", "w")
    procs[tag] = (subprocess.Popen([sys.executable, "train.py"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT), time.time())
    print(f"{time.strftime('%H:%M:%S')} start {tag}", flush=True)
    time.sleep(5)
while procs:
    for t, (pr, t0) in list(procs.items()):
        if pr.poll() is not None:
            print(f"{time.strftime('%H:%M:%S')} {t} rc={pr.returncode} {time.time()-t0:.0f}s", flush=True)
            del procs[t]
    time.sleep(20)
while ours():
    time.sleep(20)
print("ALL DONE", flush=True)
