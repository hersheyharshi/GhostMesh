import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghostmesh import ROUTERS, Config, Simulation  # noqa: E402
from ghostmesh.forecast import GhostForecaster  # noqa: E402
from ghostmesh.metrics import auc, bootstrap_ci  # noqa: E402
from ghostmesh.sim import Packet  # noqa: E402
from ghostmesh.trust import TrustModel  # noqa: E402
from server.live import LiveSession  # noqa: E402

SMALL = Config(n_nodes=30, steps=80, seed=7)


# ------------------------------------------------------------------ simulator
@pytest.mark.parametrize("router", ROUTERS)
def test_runs_and_is_deterministic(router):
    a = Simulation(SMALL, router).run()
    b = Simulation(SMALL, router).run()
    assert a["generated"] > 0
    assert 0 <= a["pdr"] <= 1
    for k in ("generated", "delivered", "dropped_adv", "lost_ghost", "lost_ttl", "data_tx"):
        assert a[k] == b[k], k


@pytest.mark.parametrize("router", ["aodv", "etx", "trust_etx", "ghostmesh"])
def test_packet_conservation(router):
    s = Simulation(SMALL, router).run()
    accounted = s["delivered"] + s["dropped_adv"] + s["lost_ghost"] + s["lost_ttl"] + s["inflight_end"]
    assert accounted == s["generated"]


def test_every_router_faces_the_same_world():
    sims = [Simulation(SMALL, r) for r in ROUTERS]
    for s in sims[1:]:
        np.testing.assert_array_equal(s.world.pos, sims[0].world.pos)
        np.testing.assert_array_equal(s.malicious, sims[0].malicious)


def test_no_attackers_means_no_adversarial_drops():
    s = Simulation(SMALL.with_(malicious_frac=0.0), "ghostmesh").run()
    assert s["dropped_adv"] == 0


def test_hops_follow_radio_links():
    sim = Simulation(SMALL, "ghostmesh")
    for _ in range(40):
        sim.step()
        for u, v, _ok in sim._step_hops:
            assert sim.world.in_range[u, v]


def test_trust_router_avoids_distrusted_node():
    sim = Simulation(SMALL.with_(malicious_frac=0.0), "trust_etx")
    sim.step()
    u = int(np.where(~sim.world.is_gateway)[0][5])
    pkt = Packet(0, u, 0, u, [u], {u})
    v = sim.router.next_hop(sim, u, pkt)
    assert v is not None
    sim.T[u, v] = 0.0                       # u now distrusts its preferred next hop
    sim.router.begin_step(sim)
    v2 = sim.router.next_hop(sim, u, Packet(1, u, 0, u, [u], {u}))
    assert v2 != v


def test_aodv_blackhole_attracts_routes():
    cfg = SMALL.with_(malicious_frac=0.3, attack_start=0, attack_mix=("blackhole",))
    s = Simulation(cfg, "aodv").run()
    assert s["dropped_adv"] > 0


def test_kill_and_compromise():
    sim = Simulation(SMALL, "ghostmesh")
    sim.step()
    i = int(np.where(~sim.world.is_gateway & ~sim.malicious)[0][0])
    sim.kill(i)
    assert not sim.world.alive[i]
    j = int(np.where(~sim.world.is_gateway & ~sim.malicious & sim.world.alive)[0][0])
    sim.compromise(j)
    assert sim.malicious[j]


# ---------------------------------------------------------------------- trust
def test_beta_trust_updates():
    t = TrustModel(3, prior=(2.0, 1.0))
    base = t.direct()[0, 1]
    t.observe(0, 1, good=False)
    assert t.direct()[0, 1] < base
    t.observe(0, 2, good=True)
    assert t.direct()[0, 2] > base


def test_asymmetric_forgetting_remembers_misbehaviour():
    sym = TrustModel(2, asymmetric=False)
    asym = TrustModel(2, asymmetric=True)
    for m in (sym, asym):
        for _ in range(5):
            m.observe(0, 1, good=False)
        for _ in range(100):
            m.decay()
    assert asym.direct()[0, 1] < sym.direct()[0, 1]


def _badmouth_setup(filtered):
    """Node 0 has good first-hand experience with honest node 4; liars 5..9 badmouth it."""
    n = 10
    t = TrustModel(n, filtered=filtered)
    malicious = np.zeros(n, bool)
    malicious[5:] = True
    nb = np.ones((n, n), bool)
    for _ in range(6):
        t.observe(0, 4, good=True)
        for j in range(1, 4):               # 0 also knows honest 1..3 and 1..3 know 4
            t.observe(0, j, good=True)
            t.observe(j, 4, good=True)
    for _ in range(5):
        t.exchange(nb, malicious, badmouth=True)
    return t.trust()[0, 4]


def test_recommendation_filter_resists_badmouthing():
    assert _badmouth_setup(filtered=True) > _badmouth_setup(filtered=False)
    assert _badmouth_setup(filtered=True) > 0.6


# ----------------------------------------------------------------- forecaster
def test_forecaster_learns_synthetic_signal():
    rng = np.random.default_rng(0)
    n, h = 120, 5
    fc = GhostForecaster(n, horizon=h, lr=0.5, rng=rng)
    death = rng.uniform(20, 400, n)                   # every node dies at some point
    nb = np.ones((n, n), bool) & ~np.eye(n, dtype=bool)
    for t in range(400):
        alive = death > t
        X = np.zeros((n, 7))
        X[:, 0] = 1
        X[:, 1] = rng.random(n)                       # uninformative
        # "stress" rises during the last few steps before death
        X[:, 3] = 0.3 * rng.random(n) + 0.7 * ((death - t) < 4)
        fc.step(t, X, alive, nb, death, np.ones(n, bool))
    p, y = np.array(fc.eval_p), np.array(fc.eval_y)
    half = len(p) // 2
    assert y[half:].sum() > 20
    assert auc(p[half:], y[half:]) > 0.8


# -------------------------------------------------------------------- metrics
def test_auc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.3
    s = y * 0.8 + rng.random(500)
    s[:50] = np.round(s[:50], 1)            # include ties
    assert abs(auc(s, y) - sk.roc_auc_score(y, s)) < 1e-9


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < m < hi


# ----------------------------------------------------------------------- live
def test_live_session_protocol():
    ls = LiveSession(seed=1, baseline="etx", steps=20)
    st = ls.static()
    assert st["type"] == "static" and len(st["pos"]) == st["n"]
    fr = ls.step()
    assert set(fr["frames"]) == {"baseline", "ghostmesh"}
    import json
    json.dumps(st), json.dumps(fr)          # must be plain-JSON serialisable for the WebSocket
    for _ in range(10):
        json.dumps(ls.step())
    victim = int(np.where(~ls.sims["ghostmesh"].world.is_gateway)[0][0])
    ls.command({"cmd": "kill", "node": victim})
    assert not ls.sims["baseline"].world.alive[victim]
    with pytest.raises(ValueError):
        LiveSession(baseline="ghostmesh")


# ------------------------------------------------------------------- analysis
def test_topology_report_bounds():
    from ghostmesh.analysis import topology_report
    sim = Simulation(SMALL, "etx")
    for _ in range(30):
        sim.step()
    r = topology_report(sim)
    assert r["components"] >= 1
    assert 0 <= r["reachable_honest_path"] <= r["reachable_any_path"] <= 1
