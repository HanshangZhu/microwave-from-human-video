"""Re-solve chosen configurations as A (solver only) and B (human-guided) with phase step logging, then render hq views."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = f"{ROOT}/Gitignore-but-important/sim"
OUT = f"{G}/pairs"
RUNS = [3, 6, 0, 1, 4, 7, 2, 5, 29, 17]
h = json.load(open(f"{G}/genh/summary.json"))["runs"]
PY = sys.executable
env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
def main():
    jobs = []
    for r in RUNS:
        c = h[r]
        for arm in ("A", "B"):
            d = f"{OUT}/{arm}/run_{r:03d}"
            cmd = [PY, f"{ROOT}/sim/solve_microwave.py", "--out", d, "--no-video", "--pot", *map(str, c["pot"]), "--place", *map(str, c["place"])]
            if arm == "B":
                cmd += ["--human", c["human"]]
            jobs.append((d, cmd))
    def run(j):
        d, cmd = j
        os.makedirs(d, exist_ok=True)
        with open(d + ".log", "w") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            return d, json.load(open(f"{d}/result.json"))["success"]
        except Exception as e:
            return d, f"crashed {e}"
    with ThreadPoolExecutor(10) as ex:
        for d, s in ex.map(run, jobs):
            print(os.path.relpath(d, G), s, flush=True)
    sys.path.insert(0, ROOT)
    from sim.render_views import render_run
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        for d, msg in ex.map(render_run, [(d, 4, "take_000", True) for d, _ in jobs]):
            print("render", os.path.relpath(d, G), msg, flush=True)


if __name__ == "__main__":
    main()
