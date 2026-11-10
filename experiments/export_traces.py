"""Record replay traces for the static dashboard (GitHub Pages needs no backend).

    python experiments/export_traces.py --seed 3

Writes ``docs/traces/seed_<k>.json``: one shared world, one run per router.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ghostmesh import LABELS, ROUTERS, Config, Simulation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[4, 13, 12])
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "docs", "traces"), exist_ok=True)
    index = []
    for seed in args.seed:
        cfg = Config(seed=seed)
        out = {"seed": seed, "labels": LABELS, "runs": {}, "summary": {}}
        for r in ROUTERS:
            sim = Simulation(cfg, r, record=True)
            s = sim.run()
            out.setdefault("static", sim.static_frame())
            out["runs"][r] = sim.trace
            out["summary"][r] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in s.items() if isinstance(v, (int, float))}
        path = os.path.join(ROOT, "docs", "traces", f"seed_{seed}.json")
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        index.append(seed)
        print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    # Seeds 4 / 13 / 12 are the median / best / worst GhostMesh-vs-OLSR-ETX gain in the benchmark.
    names = {4: "seed 4 · typical (median gain)", 13: "seed 13 · best case", 12: "seed 12 · worst case"}
    with open(os.path.join(ROOT, "docs", "traces", "index.json"), "w") as f:
        json.dump({"seeds": index, "labels": {str(s): names.get(s, f"seed {s}") for s in index}}, f)


if __name__ == "__main__":
    main()
