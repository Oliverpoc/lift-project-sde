"""Run training jobs from a job file with at most 3 parallel processes (CPU budget)."""
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent
(ROOT / "logs").mkdir(exist_ok=True)
jobs = [l.split() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
def run(a):
    tag = "_".join(a)
    with open(ROOT / "logs" / f"train_{tag}.log", "w") as f:
        r = subprocess.run([sys.executable, "train.py"] + a, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print("finished", tag, r.returncode, flush=True)
with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, jobs))
