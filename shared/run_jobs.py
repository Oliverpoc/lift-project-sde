"""Job launcher for shared: at most P concurrent train.py processes (started by this launcher).
usage: python run_jobs.py JOBFILE [P]"""
import subprocess, sys, time
from pathlib import Path
from common import ROOT, run_tag
(ROOT / "logs").mkdir(exist_ok=True)

jobfile, P = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
lines = [l.split() for l in Path(jobfile).read_text().splitlines() if l.strip() and not l.startswith("#")]
procs = {}


def reap():
    for t, (pr, t0) in list(procs.items()):
        if pr.poll() is not None:
            print(f"{time.strftime('%H:%M:%S')} {t} rc={pr.returncode} {time.time()-t0:.0f}s", flush=True)
            del procs[t]


for args in lines:
    w, ex, m, s = args[0], args[1], args[2], int(args[3]); dh = int(args[4]) if len(args) > 4 else 0
    tag = run_tag(w, ex, m, dh, s)
    if (ROOT / "runs" / tag / "model.pt").exists():
        continue
    while len(procs) >= P:
        reap(); time.sleep(10)
    f = open(ROOT / "logs" / f"{tag}.log", "w")
    procs[tag] = (subprocess.Popen([sys.executable, "train.py"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT), time.time())
    print(f"{time.strftime('%H:%M:%S')} start {tag}", flush=True)
while procs:
    reap(); time.sleep(10)
print("ALL DONE", time.strftime('%H:%M:%S'), flush=True)
