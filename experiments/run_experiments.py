"""Reproduce every number and figure in the README.

    python experiments/run_experiments.py            # full study (~15 min on 2 cores)
    python experiments/run_experiments.py --quick    # smoke version for CI (~1 min)

Writes CSV/JSON to ``results/``; ``make_figures.py`` turns them into plots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ghostmesh import ABLATIONS, ROUTERS, Config, Simulation  # noqa: E402
from ghostmesh.metrics import auc, bootstrap_ci, roc_curve  # noqa: E402

OUT = os.path.join(ROOT, "results")
SCALARS = ["pdr", "latency_mean", "latency_p95", "hops_mean", "energy_per_delivery",
           "tx_per_delivery", "adv_drop_frac", "ghost_loss_frac", "lost_ttl", "tx_to_ghost",
           "half_life", "alive_end", "detect_auc", "detect_tpr", "detect_fpr",
           "generated", "delivered", "dropped_adv", "lost_ghost", "inflight_end"]


# ----------------------------------------------------------------- workers
def _run(job):
    tag, router, seed, cfg_kw, spec_name = job
    cfg = Config(seed=seed, **cfg_kw)
    spec = ABLATIONS[spec_name] if spec_name else None
    sim = Simulation(cfg, router, spec=spec)
    s = sim.run()
    row = {k: s.get(k, np.nan) for k in SCALARS}
    row.update(tag=tag, router=spec_name or router, seed=seed, **{f"cfg_{k}": v for k, v in cfg_kw.items()})
    ts = {k: v for k, v in sim.ts.items() if v}
    return row, ts


def _forecast_job(seed):
    """Collect forecaster predictions + features from one run (router-independent)."""
    from sklearn.linear_model import LogisticRegression

    out = {}
    for gossip in (True, False):
        sim = Simulation(Config(seed=seed, fc_gossip=gossip), "etx")
        sim.run()
        f = sim.forecaster
        p, y, t = np.array(f.eval_p), np.array(f.eval_y), np.array(f.eval_t)
        X = np.vstack(f.eval_X)
        out["gossip" if gossip else "local"] = (p, y, t, X)
    p, y, t, X = out["gossip"]
    test = t >= np.median(t)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[~test], y[~test])
    pl, yl, tl, _ = out["local"]
    test_l = tl >= np.median(tl)
    scores = {
        "Gossip-learned (GhostMesh)": (p[test], y[test]),
        "Local-only learning": (pl[test_l], yl[test_l]),
        "Centralised oracle (offline LR)": (lr.predict_proba(X[test])[:, 1], y[test]),
        "Battery heuristic": (1 - X[test, 1], y[test]),
        "Stress sensor only": (X[test, 3], y[test]),
    }
    return seed, {k: (v[0].tolist(), v[1].tolist()) for k, v in scores.items()}


def pmap(fn, jobs, procs):
    if procs <= 1:
        return [fn(j) for j in jobs]
    with Pool(procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


def summarise(df: pd.DataFrame, group: str, metrics) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group, sort=False):
        r = {group: name, "n": len(g)}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].values)
            r[m], r[m + "_lo"], r[m + "_hi"] = mean, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    global OUT
    if args.quick:
        OUT = os.path.join(OUT, "quick")     # never clobber the full results
    os.makedirs(OUT, exist_ok=True)
    n_main = 3 if args.quick else 20
    n_sweep = 2 if args.quick else 10
    cfg_quick = {"steps": 120} if args.quick else {}
    t0 = time.time()
    report = {"seeds_main": n_main, "seeds_sweep": n_sweep}

    # 1) Main benchmark -------------------------------------------------------
    jobs = [("main", r, s, cfg_quick, None) for r in ROUTERS for s in range(n_main)]
    res = pmap(_run, jobs, args.procs)
    main_df = pd.DataFrame([r for r, _ in res])
    main_df.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    summ = summarise(main_df, "router", ["pdr", "latency_mean", "energy_per_delivery", "tx_per_delivery",
                                         "adv_drop_frac", "ghost_loss_frac", "alive_end", "half_life",
                                         "detect_auc", "detect_tpr", "detect_fpr"])
    summ.to_csv(os.path.join(OUT, "benchmark_summary.csv"), index=False)
    # time series (mean over seeds)
    ts_rows = []
    for (row, ts) in res:
        for i in range(len(ts["alive"])):
            ts_rows.append(dict(router=row["router"], seed=row["seed"], t=i,
                                **{k: ts[k][i] for k in ts}))
    pd.DataFrame(ts_rows).to_csv(os.path.join(OUT, "timeseries.csv"), index=False)
    # paired significance: GhostMesh vs each baseline on PDR
    gm = main_df[main_df.router == "ghostmesh"].set_index("seed")
    tests = {}
    for r in ROUTERS:
        if r == "ghostmesh":
            continue
        b = main_df[main_df.router == r].set_index("seed")
        d = (gm["pdr"] - b["pdr"]).dropna()
        stat = wilcoxon(d) if len(d) >= 5 and np.any(d != 0) else None
        m, lo, hi = bootstrap_ci(d.values)
        tests[r] = dict(mean_diff=m, ci=[lo, hi], wins=int((d > 0).sum()), n=int(len(d)),
                        p_value=float(stat.pvalue) if stat else None)
    report["pdr_vs_baselines"] = tests
    print(f"[{time.time()-t0:5.0f}s] benchmark done")

    # 2) Attack-intensity sweep -----------------------------------------------
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
    jobs = [("sweep", r, s, {**cfg_quick, "malicious_frac": f}, None)
            for f in fracs for r in ROUTERS for s in range(n_sweep)]
    res = pmap(_run, jobs, args.procs)
    sweep = pd.DataFrame([r for r, _ in res])
    sweep.to_csv(os.path.join(OUT, "attack_sweep.csv"), index=False)
    print(f"[{time.time()-t0:5.0f}s] attack sweep done")

    # 3) Ablation ------------------------------------------------------------
    jobs = [("ablation", "ghostmesh", s, cfg_quick, name) for name in ABLATIONS for s in range(n_main)]
    res = pmap(_run, jobs, args.procs)
    abl = pd.DataFrame([r for r, _ in res])
    abl.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    full = abl[abl.router == "ghostmesh"].set_index("seed")
    abl_rows = []
    for name in ABLATIONS:
        g = abl[abl.router == name].set_index("seed")
        row = {"variant": name}
        for m in ["pdr", "adv_drop_frac", "ghost_loss_frac", "detect_auc"]:
            mean, lo, hi = bootstrap_ci(g[m].values)
            dm, dlo, dhi = bootstrap_ci((g[m] - full[m]).values)
            row.update({m: mean, f"{m}_delta": dm, f"{m}_delta_lo": dlo, f"{m}_delta_hi": dhi})
        abl_rows.append(row)
    pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "ablation_summary.csv"), index=False)
    print(f"[{time.time()-t0:5.0f}s] ablation done")

    # 4) Ghost forecaster ----------------------------------------------------
    fc = pmap(_forecast_job, list(range(3 if args.quick else 10)), args.procs)
    fc_summary, roc = {}, {}
    names = list(fc[0][1].keys())
    for name in names:
        per_seed = [auc(*map(np.array, d[name])) for _, d in fc]
        pooled_p = np.concatenate([d[name][0] for _, d in fc])
        pooled_y = np.concatenate([d[name][1] for _, d in fc]).astype(bool)
        m, lo, hi = bootstrap_ci(per_seed)
        fc_summary[name] = dict(auc=m, ci=[lo, hi], pooled_auc=auc(pooled_p, pooled_y),
                                positives=int(pooled_y.sum()), samples=int(len(pooled_y)))
        fpr, tpr = roc_curve(pooled_p, pooled_y)
        roc[name] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
    report["forecaster"] = fc_summary
    with open(os.path.join(OUT, "forecaster_roc.json"), "w") as f:
        json.dump(roc, f)
    print(f"[{time.time()-t0:5.0f}s] forecaster done")

    report["runtime_s"] = round(time.time() - t0, 1)
    report["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in Config().to_dict().items()}
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(summ[["router", "pdr", "pdr_lo", "pdr_hi", "energy_per_delivery", "adv_drop_frac", "alive_end"]].round(3))
    print(json.dumps(tests, indent=1, default=float))


if __name__ == "__main__":
    main()
