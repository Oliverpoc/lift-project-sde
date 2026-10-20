"""Run lines of a job file in parallel (at most P processes).
usage: python run_jobs.py JOBFILE [P] [SCRIPT]
Each line: arguments for SCRIPT (default train.py). Skips train jobs whose runs/<tag>/model.pt exists."""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, run_tag
(ROOT / "logs").mkdir(exist_ok=True)

jobfile, P = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
script = sys.argv[3] if len(sys.argv) > 3 else "train.py"
lines = [l.split() for l in Path(jobfile).read_text().splitlines() if l.strip() and not l.startswith("#")]


def run(args):
    if script == "train.py":
        w, ex, m, s = args[0], args[1], args[2], int(args[3]); dh = int(args[4]) if len(args) > 4 else 0
        tag = run_tag(w, ex, m, dh, s)
        if (ROOT / "runs" / tag / "model.pt").exists():
            return tag, 0, 0.0
    else:
        tag = "eval_" + "_".join(args)
    t0 = time.time()
    with open(ROOT / "logs" / f"{tag}.log", "w") as f:
        r = subprocess.run([sys.executable, script] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"{time.strftime('%H:%M:%S')} {tag} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
    return tag, r.returncode, time.time() - t0


with ThreadPoolExecutor(P) as ex:
    res = list(ex.map(run, lines))
bad = [r for r in res if r[1] != 0]
print("ALL DONE", len(res), "failed:", bad, flush=True)
