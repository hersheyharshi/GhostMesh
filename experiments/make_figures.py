"""Turn ``results/`` into the figures in ``docs/figures`` and ``docs/headline.json``."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ghostmesh import LABELS, ROUTERS  # noqa: E402
from ghostmesh.metrics import bootstrap_ci  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

# Colour follows the protocol everywhere (validated categorical palette, light mode).
COLOR = {"ghostmesh": "#2a78d6", "trust_etx": "#eb6834", "etx": "#1baf7a",
         "aodv": "#eda100", "flooding": "#e87ba4"}
MARK = {"ghostmesh": "o", "trust_etx": "s", "etx": "^", "aodv": "D", "flooding": "v"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_benchmark(df):
    metrics = [("pdr", "SOS delivery ratio", "higher is better", 100, "%"),
               ("energy_per_delivery", "Energy per delivered packet", "lower is better", 1000, " mJ*"),
               ("alive_end", "Relays alive at the end", "higher is better", 100, "%")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"wspace": 0.55})
    order = ROUTERS[::-1]
    for ax, (m, title, sub, scale, unit) in zip(axes, metrics):
        top = max(bootstrap_ci(df[df.router == r][m].values)[2] for r in order) * scale
        ax.set_xlim(0, top * 1.32)
        for i, r in enumerate(order):
            mean, lo, hi = bootstrap_ci(df[df.router == r][m].values)
            ax.barh(i, mean * scale, color=COLOR[r], height=0.62, edgecolor=SURF, linewidth=2)
            ax.errorbar(mean * scale, i, xerr=[[(mean - lo) * scale], [(hi - mean) * scale]],
                        color=INK, capsize=3, lw=1.1)
            ax.text(hi * scale, i, f"  {mean*scale:.1f}{'%' if unit == '%' else ''}",
                    va="center", color=INK, fontsize=9.5)
        ax.set_yticks(range(len(order)), [LABELS[r] for r in order])
        ax.set_title(f"{title}\n", loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("energy units ×1000 (relative)")
    axes[0].set_xlabel("% of generated SOS packets")
    axes[2].set_xlabel("% of non-gateway nodes")
    fig.suptitle(f"Main benchmark — 50 nodes, 20% insider attackers, spreading hazard (n={df.seed.nunique()} seeds, 95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=11, color=INK2, y=1.06)
    save(fig, "benchmark.png")


def fig_fates(df):
    fates = [("delivered", "Delivered", "#2a78d6"), ("dropped_adv", "Swallowed by attacker", "#e34948"),
             ("lost_ghost", "Lost with dying node", "#6250d6"), ("other", "Expired / never routed", "#a3a29c")]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    order = ROUTERS[::-1]
    for i, r in enumerate(order):
        g = df[df.router == r]
        gen = g.generated.sum()
        vals = {"delivered": g.delivered.sum() / gen, "dropped_adv": g.dropped_adv.sum() / gen,
                "lost_ghost": g.lost_ghost.sum() / gen}
        vals["other"] = 1 - sum(vals.values())
        left = 0
        for key, lab, col in fates:
            v = vals[key]
            ax.barh(i, v * 100, left=left, color=col, height=0.62, edgecolor=SURF, linewidth=2,
                    label=lab if i == 0 else None)
            if v > 0.06:
                ax.text(left + v * 50, i, f"{v*100:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += v * 100
    ax.set_yticks(range(len(order)), [LABELS[r] for r in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of generated SOS packets")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where every SOS packet ended up")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=9)
    save(fig, "packet_fates.png")


def fig_sweep(sw):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (m, title, scale) in zip(axes, [("pdr", "Delivery ratio vs. share of attackers", 100),
                                             ("energy_per_delivery", "Energy per delivered packet", 1000)]):
        for r in ROUTERS:
            g = sw[sw.router == r].groupby("cfg_malicious_frac")[m]
            x = g.mean().index.values * 100
            mean = g.mean().values * scale
            se = g.sem().values * scale * 1.96
            ax.plot(x, mean, marker=MARK[r], ms=7, color=COLOR[r], label=LABELS[r],
                    markeredgecolor=SURF, markeredgewidth=1.5)
            ax.fill_between(x, mean - se, mean + se, color=COLOR[r], alpha=0.07, lw=0)
            if m == "pdr":
                ax.annotate(LABELS[r], (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK)
        ax.set_xlabel("insider attackers (% of relays)")
        ax.set_title(title)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlim(-2, 50)
    axes[0].set_ylabel("% delivered")
    axes[1].set_ylabel("energy units ×1000")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=9, ncol=2)
    axes[1].set_xlim(-2, 42)
    save(fig, "attack_sweep.png")


def fig_ablation(ab):
    ab = ab[ab.variant != "ghostmesh"].copy()
    ab = ab.sort_values("pdr_delta")
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(ab))
    d = ab.pdr_delta.values * 100
    lo, hi = ab.pdr_delta_lo.values * 100, ab.pdr_delta_hi.values * 100
    cols = ["#e34948" if h < 0 else "#a3a29c" for h in hi]
    ax.barh(y, d, color=cols, height=0.6, edgecolor=SURF, linewidth=2)
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="none", color=INK, capsize=3, lw=1.1)
    for yi, (v, h, l) in enumerate(zip(d, hi, lo)):
        ax.text(min(l, 0) - 0.4, yi, f"{v:+.1f} pts", ha="right", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, ab.variant.values)
    ax.set_xlim(min(lo.min(), 0) - 2.5, max(hi.max(), 0) + 1)
    ax.set_xlabel("change in delivery ratio vs. full GhostMesh (percentage points, paired by seed)")
    ax.set_title("Ablation — what each component is worth\n")
    ax.text(0, 1.02, "red = removal hurts with 95% confidence · grey = interval crosses zero",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "ablation.png")


def fig_timeseries(ts):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for r in ROUTERS:
        g = ts[ts.router == r].groupby("t")
        pdr = g.pdr_window.mean().rolling(9, center=True, min_periods=1).mean() * 100
        axes[0].plot(pdr.index, pdr.values, color=COLOR[r], label=LABELS[r])
        axes[1].plot(g.alive.mean().index, g.alive.mean().values * 100, color=COLOR[r], label=LABELS[r])
    axes[0].axvspan(0, 20, color="#a3a29c", alpha=0.12, lw=0)
    axes[0].text(2, 8, "attacks\nstart t=20", fontsize=8.5, color=INK2)
    axes[0].set_title("Delivery ratio over time (30-step window)")
    axes[0].set_ylabel("% delivered")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Network lifetime")
    axes[1].set_ylabel("% relays alive")
    axes[1].set_xlim(0, 300)
    axes[1].legend(fontsize=9, loc="lower left")
    for ax in axes:
        ax.set_xlabel("time step")
    axes[0].legend(fontsize=9, ncol=3, loc="upper right")
    axes[0].set_ylim(0, 115)
    save(fig, "timeseries.png")


def fig_trust(ts):
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    for r, ls in [("trust_etx", "-"), ("ghostmesh", "-")]:
        g = ts[ts.router == r].groupby("t")
        ax.plot(g.rep_malicious.mean().index, g.rep_malicious.mean() * 100, color=COLOR[r], ls="-",
                label=f"{LABELS[r]} · attackers")
        ax.plot(g.rep_honest.mean().index, g.rep_honest.mean() * 100, color=COLOR[r], ls=(0, (2, 2)),
                label=f"{LABELS[r]} · honest relays")
    ax.axhline(35, color=INK2, lw=1, ls=":")
    ax.text(302, 35, "exclusion\nthreshold", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 330)
    ax.set_ylim(0, 100)
    ax.set_xlabel("time step")
    ax.set_ylabel("mean reputation (%)")
    ax.set_title("Reputation separates attackers from honest relays — only with GhostMesh's defences")
    ax.legend(fontsize=9, ncol=2, loc="lower left")
    save(fig, "trust.png")


def fig_roc(roc, rep):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for (name, c), col in zip(roc.items(), cols):
        a = rep["forecaster"][name]["auc"]
        ax.plot(c["fpr"], c["tpr"], color=col, label=f"{name}  (AUC {a:.2f})")
    ax.plot([0, 1], [0, 1], color="#a3a29c", lw=1, ls=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Ghost forecaster: will this node die\nwithin 12 steps?")
    ax.legend(fontsize=8.2, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "forecaster_roc.png")


def headline(summ, rep):
    s = summ.set_index("router")
    gm, te, etx, fl = s.loc["ghostmesh"], s.loc["trust_etx"], s.loc["etx"], s.loc["flooding"]
    stats = [
        {"value": f"+{(gm.pdr - etx.pdr)*100:.0f} pts",
         "label": f"SOS delivery vs. OLSR-ETX ({etx.pdr*100:.0f}% → {gm.pdr*100:.0f}%)"},
        {"value": f"−{(1 - gm.adv_drop_frac / etx.adv_drop_frac)*100:.0f}%",
         "label": "packets swallowed by insider attackers"},
        {"value": f"−{(1 - gm.energy_per_delivery / fl.energy_per_delivery)*100:.0f}%",
         "label": f"energy per delivery vs. flooding, at {'similar' if abs(gm.pdr-fl.pdr) < 0.03 else 'comparable'} delivery"},
    ]
    with open(os.path.join(ROOT, "docs", "headline.json"), "w") as f:
        json.dump({"stats": stats, "seeds": int(s.loc["ghostmesh", "n"])}, f, indent=1)
    print("headline:", stats)


def main():
    df = pd.read_csv(os.path.join(RES, "benchmark.csv"))
    summ = pd.read_csv(os.path.join(RES, "benchmark_summary.csv"))
    rep = json.load(open(os.path.join(RES, "report.json")))
    fig_benchmark(df)
    fig_fates(df)
    ts = pd.read_csv(os.path.join(RES, "timeseries.csv"))
    fig_timeseries(ts)
    fig_trust(ts)
    fig_sweep(pd.read_csv(os.path.join(RES, "attack_sweep.csv")))
    fig_ablation(pd.read_csv(os.path.join(RES, "ablation_summary.csv")))
    fig_roc(json.load(open(os.path.join(RES, "forecaster_roc.json"))), rep)
    headline(summ, rep)


if __name__ == "__main__":
    main()
