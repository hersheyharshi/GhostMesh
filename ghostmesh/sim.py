"""Discrete-time simulator for an emergency mesh under failures and attacks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config, RouterSpec
from .forecast import GhostForecaster
from .routers import Router, make_router
from .trust import TrustModel
from .world import World


@dataclass(eq=False)
class Packet:
    pid: int
    src: int
    created: int
    holder: int
    path: list = field(default_factory=list)
    visited: set = field(default_factory=set)
    retries: int = 0
    route: list | None = None
    moved_at: int = -1


@dataclass(eq=False)
class FloodPacket:
    pid: int
    src: int
    created: int
    holders: list
    seen: set


class Simulation:
    def __init__(self, cfg: Config, router: str | Router = "ghostmesh",
                 spec: RouterSpec | None = None, record: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Separate stream for the world so every router faces the *same* disaster.
        self.world = World(cfg, np.random.default_rng(cfg.seed + 10_000))
        self.n = cfg.n_nodes
        self.router = router if isinstance(router, Router) else make_router(router, spec)
        self.t = 0

        # ------------------------------------------------------- adversaries
        n = self.n
        arng = np.random.default_rng(cfg.seed + 20_000)
        cand = np.arange(cfg.n_gateways, n)
        k = int(round(cfg.malicious_frac * len(cand)))
        mal = arng.choice(cand, size=k, replace=False) if k else np.array([], int)
        self.malicious = np.zeros(n, bool)
        self.malicious[mal] = True
        self.attack_type = np.array([""] * n, dtype=object)
        for i, m in enumerate(sorted(mal)):
            self.attack_type[m] = cfg.attack_mix[i % len(cfg.attack_mix)]
        self.attacking = np.zeros(n, bool)
        self.lying = np.zeros(n, bool)

        # ------------------------------------------------------- protocol state
        spec_ = getattr(self.router, "spec", None)
        self.trust = None
        if self.router.uses_trust:
            self.trust = TrustModel(n, cfg.trust_prior, spec_.asymmetric_forgetting,
                                    spec_.filtered_recs, cfg.rec_discount,
                                    np.random.default_rng(cfg.seed + 30_000))
        self.T = None
        self.forecaster = GhostForecaster(n, cfg.horizon, cfg.fc_lr, cfg.fc_pos_weight, cfg.fc_gossip)
        self.tau = np.zeros((n, n))
        self.blacklist = np.zeros((n, n), int)
        self.known_alive = np.ones(n, bool)
        self.adv_etx = np.full((n, n), np.inf)
        self.adv_risk = np.zeros(n)
        self.adv_queue = np.zeros(n)
        self.watches: dict[int, list] = {}

        # ------------------------------------------------------- traffic & stats
        self.packets: list[Packet] = []
        self.floods: list[FloodPacket] = []
        self.next_pid = 0
        self.tx = np.zeros(n)
        self.rx = np.zeros(n)
        self.cap = np.zeros(n, int)
        self.stats = dict(generated=0, delivered=0, lost_ttl=0, lost_ghost=0,
                          dropped_adv=0, data_tx=0, ctrl_tx=0, energy=0.0, tx_to_ghost=0)
        self.latencies: list[int] = []
        self.hops: list[int] = []
        self.ts = {k: [] for k in ("alive", "generated", "delivered", "pdr_window", "energy",
                                     "rep_malicious", "rep_honest")}
        self._win = []
        self.record = record
        self.trace: list[dict] = []
        self._step_hops: list = []
        self._step_drops: list = []

    # ================================================================ helpers
    def control_flood(self, origin: int):
        """Charge a network-wide route-request flood (AODV discovery)."""
        reach = self.world.alive & ~self.world.is_gateway
        self.stats["ctrl_tx"] += int(reach.sum())
        self.tx[reach] += self.cfg.e_ctrl / self.cfg.e_tx

    def _advertise(self):
        """Build the link-state database every node receives (possibly forged)."""
        w = self.world
        with np.errstate(divide="ignore"):
            etx = np.where(w.pdr > 0.05, 1.0 / np.maximum(w.pdr, 1e-9), np.inf)
        etx[~w.in_range] = np.inf
        if self.cfg.lie_links and self.lying.any():
            L = self.lying
            fake = w.in_range & (L[:, None] | L[None, :])
            etx[fake] = 1.0
        self.adv_etx = etx

    def _verify_claims(self):
        """Advertisement verification: compare a neighbour's claimed link ETX with
        what I measure on the same link (probe loss, with estimation noise)."""
        w = self.world
        meas = np.clip(w.pdr + self.rng.normal(0, 0.04, w.pdr.shape), 0.02, 1.0)
        meas_etx = 1.0 / meas
        claim = self.adv_etx.T            # claim[u, v] = what v advertises about link (v, u)
        live = w.in_range & w.alive[:, None] & w.alive[None, :] & (meas > 0.2)
        honest_obs = ~self.malicious[:, None]
        bad = live & honest_obs & (meas_etx > 1.5 * claim)
        if bad.any():
            self.trust.beta += 0.3 * bad
            for u, v in zip(*np.where(bad)):
                self.router.on_misbehaviour(self, int(u), int(v))

    def _update_attackers(self):
        cfg, t = self.cfg, self.t
        on = t >= cfg.attack_start
        phase_on = (t // max(cfg.onoff_period // 2, 1)) % 2 == 0
        act = self.malicious & on & self.world.alive
        act &= (self.attack_type != "onoff") | phase_on
        self.attacking = act
        self.lying = act.copy()

    def _drops(self, u: int) -> bool:
        if not self.attacking[u]:
            return False
        if self.attack_type[u] == "grayhole":
            return self.rng.random() < self.cfg.grayhole_drop
        return True

    def _watch(self, observer, target, pkt):
        if self.trust is None or self.malicious[observer]:
            return
        self.watches.setdefault(pkt.pid, []).append([observer, target, self.t + self.cfg.watch_deadline, 0])

    def _overheard(self, pkt, u):
        ws = self.watches.get(pkt.pid)
        if not ws:
            return
        keep = []
        for obs, tgt, dl, ext in ws:
            if tgt == u and self.world.alive[obs] and self.rng.random() < min(1.0, self.world.pdr[obs, u] + 0.15):
                self.trust.observe(obs, u, True)
            else:
                keep.append([obs, tgt, dl, ext])
        if keep:
            self.watches[pkt.pid] = keep
        else:
            del self.watches[pkt.pid]

    def _expire_watches(self):
        if self.trust is None:
            return
        t = self.t
        held = {p.pid: p.holder for p in self.packets}
        for pid in list(self.watches):
            keep = []
            for obs, tgt, dl, ext in self.watches[pid]:
                if dl > t:
                    keep.append([obs, tgt, dl, ext])
                    continue
                # A node with no route broadcasts a HOLD notice (store-carry-forward).
                # Attackers mimic it to stall detection, so extensions are capped.
                holding = held.get(pid) == tgt or self.malicious[tgt]
                if holding and self.world.alive[tgt] and ext < 2:
                    keep.append([obs, tgt, t + self.cfg.watch_deadline, ext + 1])
                    continue
                # Channel-aware evidence: silence on a lossy link says little.
                q = min(1.0, self.world.pdr[obs, tgt] + 0.15) if self.trust.asymmetric else 1.0
                self.trust.observe(obs, tgt, False, weight=q)
                self.router.on_misbehaviour(self, obs, tgt)
            if keep:
                self.watches[pid] = keep
            else:
                del self.watches[pid]

    # ================================================================ traffic
    def _generate(self):
        cfg, w = self.cfg, self.world
        k = self.rng.poisson(cfg.traffic_rate)
        srcs = np.where(w.alive & ~w.is_gateway & ~self.malicious)[0]
        if len(srcs) == 0:
            return
        for s in self.rng.choice(srcs, size=k):
            s = int(s)
            if self.router.flooding:
                self.floods.append(FloodPacket(self.next_pid, s, self.t, [s], {s}))
            else:
                self.packets.append(Packet(self.next_pid, s, self.t, s, [s], {s}))
            self.next_pid += 1
            self.stats["generated"] += 1
            self._new_gen += 1

    def _deliver(self, created, hops, pkt=None):
        self.stats["delivered"] += 1
        self._new_del += 1
        self.latencies.append(self.t - created + 1)
        self.hops.append(hops)
        if pkt is not None:
            self.router.on_delivered(self, pkt)

    def _forward_unicast(self):
        w, cfg = self.world, self.cfg
        by_holder: dict[int, list[Packet]] = {}
        for p in self.packets:
            if p.moved_at < self.t:
                by_holder.setdefault(p.holder, []).append(p)
        order = list(by_holder)
        self.rng.shuffle(order)
        gone: set[int] = set()
        for u in order:
            if not w.alive[u]:
                continue
            for pkt in by_holder[u]:
                if self._drops(u):
                    gone.add(pkt.pid)
                    self.stats["dropped_adv"] += 1
                    self._step_drops.append(u)
                    continue
                if self.cap[u] >= cfg.node_capacity:
                    break
                v = self.router.next_hop(self, u, pkt)
                if v is None:
                    continue                      # store-carry-forward: wait for a route
                self.cap[u] += 1
                self.tx[u] += 1
                self.stats["data_tx"] += 1
                if self.trust is not None:
                    self._overheard(pkt, u)
                if not w.alive[v]:
                    self.stats["tx_to_ghost"] += 1
                ok = w.alive[v] and self.rng.random() < w.pdr[u, v]
                self._step_hops.append((u, int(v), bool(ok)))
                if not ok:
                    pkt.retries += 1
                    if pkt.retries >= cfg.max_retries:
                        pkt.retries = 0
                        self.blacklist[u, v] = self.t + 15
                        self.router.on_link_fail(self, u, v, pkt)
                    continue
                self.rx[v] += 1
                pkt.retries = 0
                pkt.holder = int(v)
                pkt.path.append(int(v))
                pkt.visited.add(int(v))
                pkt.moved_at = self.t
                if w.is_gateway[v]:
                    gone.add(pkt.pid)
                    self._deliver(pkt.created, len(pkt.path) - 1, pkt)
                else:
                    self._watch(u, int(v), pkt)
        if gone:
            self.packets = [p for p in self.packets if p.pid not in gone]

    def _forward_flood(self):
        w, cfg = self.world, self.cfg
        done = set()
        for fp in self.floods:
            nxt = []
            for u in fp.holders:
                if not w.alive[u] or self.cap[u] >= cfg.node_capacity:
                    if w.alive[u]:
                        nxt.append(u)           # retry next step
                    continue
                if self._drops(u):
                    self.stats["absorbed_copies"] = self.stats.get("absorbed_copies", 0) + 1
                    self._step_drops.append(u)
                    continue
                self.cap[u] += 1
                self.tx[u] += 1
                self.stats["data_tx"] += 1
                nb = np.where(w.in_range[u] & w.alive)[0]
                got = nb[self.rng.random(len(nb)) < w.pdr[u, nb]]
                for v in got:
                    v = int(v)
                    self.rx[v] += 1
                    if v in fp.seen:
                        continue
                    fp.seen.add(v)
                    self._step_hops.append((u, v, True))
                    if w.is_gateway[v]:
                        if fp.pid not in done:
                            done.add(fp.pid)
                            self._deliver(fp.created, 0)
                    else:
                        nxt.append(v)
            fp.holders = nxt
        self.floods = [f for f in self.floods if f.pid not in done and f.holders]

    # ================================================================ main loop
    def step(self):
        cfg, w, t = self.cfg, self.world, self.t
        self._new_gen = self._new_del = 0
        self._step_hops, self._step_drops = [], []
        self.tx[:] = 0
        self.rx[:] = 0
        self.cap[:] = 0

        w.update_links()
        self.known_alive = w.known_alive(t)
        self._update_attackers()
        self._advertise()

        # Ghost forecaster on HELLO telemetry; every node advertises its own risk.
        stress_obs = w.stress() + self.rng.normal(0, 0.12, self.n)
        nb = w.in_range & w.alive[None, :] & w.alive[:, None]
        X = self.forecaster.features(w.battery, stress_obs, w.alive, w.in_range, w.pdr)
        p = self.forecaster.step(t, X, w.alive, nb, w.death_time, ~w.is_gateway)
        risk = p.copy()
        risk[w.is_gateway] = 0.0
        risk[self.lying] = 0.02
        self.adv_risk = risk

        if self.trust is not None:
            self.trust.decay()
            if self.router.spec.verify_claims:
                self._verify_claims()
            if t % cfg.rec_period == 0:
                self.trust.exchange(nb, self.malicious, cfg.badmouth)
            self.T = self.trust.trust()
        self.tau *= 0.95

        self.adv_queue = np.bincount([p.holder for p in self.packets], minlength=self.n).astype(float)
        self.router.begin_step(self)
        self._generate()
        if self.router.flooding:
            self._forward_flood()
        else:
            self._forward_unicast()
        self._expire_watches()

        # TTL expiry
        if self.router.flooding:
            keep = [f for f in self.floods if t - f.created < cfg.ttl]
            self.stats["lost_ttl"] += len(self.floods) - len(keep)
            self.floods = keep
        else:
            keep = [p for p in self.packets if t - p.created < cfg.ttl]
            self.stats["lost_ttl"] += len(self.packets) - len(keep)
            self.packets = keep

        new_dead = w.step_hazards(t, self.tx, self.rx)
        self.stats["energy"] += float(cfg.e_tx * self.tx.sum() + cfg.e_rx * self.rx.sum())
        if len(new_dead) and not self.router.flooding:
            dead = set(int(i) for i in new_dead)
            lost = [p for p in self.packets if p.holder in dead]
            self.stats["lost_ghost"] += len(lost)
            if lost:
                ids = {p.pid for p in lost}
                self.packets = [p for p in self.packets if p.pid not in ids]

        self._win.append((self._new_gen, self._new_del))
        self._win = self._win[-30:]
        g = sum(a for a, _ in self._win)
        d = sum(b for _, b in self._win)
        self.ts["alive"].append(float(w.alive[~w.is_gateway].mean()))
        self.ts["generated"].append(self.stats["generated"])
        self.ts["delivered"].append(self.stats["delivered"])
        self.ts["pdr_window"].append(d / g if g else np.nan)
        self.ts["energy"].append(self.stats["energy"])
        rep = self.reputation()
        if rep is not None:
            honest = ~self.malicious & ~w.is_gateway
            self.ts["rep_malicious"].append(float(np.nanmean(rep[self.malicious])) if self.malicious.any() and np.isfinite(rep[self.malicious]).any() else np.nan)
            self.ts["rep_honest"].append(float(np.nanmean(rep[honest])) if np.isfinite(rep[honest]).any() else np.nan)
        if self.record:
            self._record_frame()
        self.t += 1

    def run(self):
        for _ in range(self.cfg.steps):
            self.step()
        return self.summary()

    # ================================================================ metrics
    def reputation(self) -> np.ndarray | None:
        """Evidence-weighted mean trust that honest live nodes place in each node.

        Observers with no evidence about a node (still at the prior) are ignored,
        so a node's reputation reflects the nodes that actually interacted with it.
        """
        if self.T is None:
            return None
        tr = self.trust
        obs = self.world.alive & ~self.malicious & ~self.world.is_gateway
        ev = (tr.alpha - tr.a0) + (tr.beta - tr.b0) + tr.alpha_r + tr.beta_r
        np.fill_diagonal(ev, 0.0)
        W = ev[obs]
        tot = W.sum(0)
        rep = (W * self.T[obs]).sum(0) / np.maximum(tot, 1e-9)
        return np.where(tot > 0.5, rep, np.nan)

    def summary(self) -> dict:
        s = dict(self.stats)
        g = max(s["generated"], 1)
        s["router"] = self.router.name
        s["seed"] = self.cfg.seed
        s["pdr"] = s["delivered"] / g
        s["latency_mean"] = float(np.mean(self.latencies)) if self.latencies else np.nan
        s["latency_p95"] = float(np.percentile(self.latencies, 95)) if self.latencies else np.nan
        s["hops_mean"] = float(np.mean(self.hops)) if self.hops else np.nan
        s["energy_per_delivery"] = s["energy"] / max(s["delivered"], 1)
        s["tx_per_delivery"] = (s["data_tx"] + s["ctrl_tx"]) / max(s["delivered"], 1)
        s["adv_drop_frac"] = s["dropped_adv"] / g
        s["ghost_loss_frac"] = s["lost_ghost"] / g
        s["inflight_end"] = len(self.packets) + len(self.floods)
        alive = np.array(self.ts["alive"])
        below = np.where(alive < 0.5)[0]
        s["half_life"] = int(below[0]) if len(below) else self.cfg.steps
        s["alive_end"] = float(alive[-1])
        rep = self.reputation()
        if rep is not None and self.malicious.any():
            from .metrics import auc
            honest = ~self.malicious & ~self.world.is_gateway
            rep = np.nan_to_num(rep, nan=self.cfg.trust_prior[0] / sum(self.cfg.trust_prior))
            sel = self.malicious | honest
            s["detect_auc"] = auc(-rep[sel], self.malicious[sel])
            th = 0.4
            alive = self.world.alive
            s["detect_tpr"] = float(np.mean(rep[self.malicious & alive] < th)) if (self.malicious & alive).any() else np.nan
            s["detect_fpr"] = float(np.mean(rep[honest & alive] < th))
        return s

    # ================================================================ live control
    def kill(self, i: int):
        """Destroy node i now (operator action in the live dashboard)."""
        w = self.world
        if w.is_gateway[i] or not w.alive[i]:
            return
        w.alive[i] = False
        w.death_time[i] = self.t
        w.death_cause[i] = "manual"
        lost = [p for p in self.packets if p.holder == i]
        self.stats["lost_ghost"] += len(lost)
        self.packets = [p for p in self.packets if p.holder != i]

    def compromise(self, i: int, kind: str = "blackhole"):
        """Turn node i into an attacker from the next step on."""
        if self.world.is_gateway[i]:
            return
        self.malicious[i] = True
        self.attack_type[i] = kind

    # ================================================================ tracing
    def static_frame(self) -> dict:
        w = self.world
        return dict(
            n=self.n, steps=self.cfg.steps, radio_range=self.cfg.radio_range,
            pos=np.round(w.pos, 4).tolist(), gateways=w.gateways.tolist(),
            malicious=np.where(self.malicious)[0].tolist(),
            attack_type={int(i): self.attack_type[i] for i in np.where(self.malicious)[0]},
            edges=w.edges.tolist(), front_origin=np.round(w.front_origin, 4).tolist(),
            router=self.router.name,
        )

    def _record_frame(self):
        self.trace.append(self.frame())

    def frame(self) -> dict:
        w = self.world
        e = w.edges
        rep = self.reputation()
        bat = np.where(np.isfinite(w.battery), np.clip(w.battery, 0, 1), 1.0)
        s = self.stats
        return dict(
            t=self.t,
            dead=np.where(~w.alive)[0].tolist(),
            ghosts=np.where(~w.alive & self.known_alive)[0].tolist(),
            attackers=np.where(self.malicious)[0].tolist(),
            battery=np.round(bat * 100).astype(int).tolist(),
            risk=np.round(self.adv_risk * 100).astype(int).tolist(),
            rep=None if rep is None else np.round(np.nan_to_num(rep, nan=-0.01) * 100).astype(int).tolist(),
            q=np.round(w.pdr[e[:, 0], e[:, 1]] * 99).astype(int).tolist(),
            front=round(float(w.front_radius), 4),
            hops=[[a, b, int(ok)] for a, b, ok in self._step_hops],
            drops=list(map(int, self._step_drops)),
            gen=s["generated"], dlv=s["delivered"], adv=s["dropped_adv"],
            ghost=s["lost_ghost"], ttl=s["lost_ttl"], energy=round(s["energy"], 3),
            inflight=len(self.packets) + len(self.floods),
        )
