"""Routing protocols compared in the benchmark.

All link-state routers route *hop by hop*: the node currently holding a packet
recomputes a least-cost path to the nearest gateway (anycast, via a virtual
super-sink) from its **own** view of the network. Views differ in what the edge
cost is made of:

=================  =========================================================
Router             cost of edge (x -> y) as seen by node u
=================  =========================================================
``flooding``       n/a — every node rebroadcasts once (epidemic baseline)
``aodv``           hop count, reactive discovery, cached source routes
``etx``            ETX(x,y) = 1 / advertised delivery ratio  (OLSR-ETX style)
``trust_etx``      ETX(x,y) / T_u(y)^gamma, with low-trust exclusion
``ghostmesh``      ETX(x,y) / T_u(y)^gamma * (1 + kappa*risk(y)) / (1 + rho*tau(x,y))
=================  =========================================================

``T_u`` is u's Bayesian trust, ``risk`` the advertised ghost-forecast and
``tau`` stigmergic pheromone deposited by acknowledged deliveries.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .config import RouterSpec

INF = np.inf


class Router:
    name = "base"
    flooding = False
    uses_trust = False
    spec: RouterSpec = RouterSpec(use_trust=False, use_forecast=False, use_pheromone=False)

    def begin_step(self, sim):
        pass

    def next_hop(self, sim, u: int, pkt) -> int | None:
        raise NotImplementedError

    def on_delivered(self, sim, pkt):
        pass

    def on_link_fail(self, sim, u: int, v: int, pkt):
        pass

    def on_misbehaviour(self, sim, observer: int, target: int):
        pass


# --------------------------------------------------------------------------- #
class FloodingRouter(Router):
    name = "flooding"
    flooding = True


# --------------------------------------------------------------------------- #
class AODVRouter(Router):
    """Reactive routing with route caching and local repair (AODV-like).

    Lying attackers answer route requests claiming a one-hop route to a gateway,
    the classic AODV black-hole attack.
    """
    name = "aodv"

    def __init__(self):
        self.cache: dict[int, list[int]] = {}

    def _discover(self, sim, u: int) -> list[int] | None:
        sim.control_flood(u)
        n = sim.n
        known = sim.known_alive
        adj = (sim.world.in_range & known[None, :] & known[:, None])
        adj[u] &= sim.blacklist[u] <= sim.t
        # BFS from u over hop count
        prev = np.full(n, -1)
        dist = np.full(n, np.inf)
        dist[u] = 0
        q = deque([u])
        while q:
            x = q.popleft()
            if sim.world.is_gateway[x] and x != u:
                continue
            for y in np.where(adj[x] & np.isinf(dist))[0]:
                dist[y] = dist[x] + 1
                prev[y] = x
                q.append(y)
        gws = sim.world.gateways
        best_gw = gws[np.argmin(dist[gws])]
        best_len = dist[best_gw]
        target = best_gw if np.isfinite(best_len) else None
        # Black holes reply "I am one hop from a gateway" and win the race on ties.
        liars = np.where(sim.lying & np.isfinite(dist))[0]
        liars = liars[liars != u]
        if len(liars):
            m = liars[np.argmin(dist[liars])]
            if dist[m] + 1 <= best_len:
                target = m
        if target is None:
            return None
        path = [int(target)]
        while path[-1] != u:
            path.append(int(prev[path[-1]]))
        return path[::-1]

    def next_hop(self, sim, u, pkt):
        route = pkt.route
        if route is None or u not in route or route[-1] == u:
            cached = self.cache.get(u)
            route = cached if cached and cached[0] == u else self._discover(sim, u)
            if route is None:
                return None
            self.cache[u] = route
            pkt.route = route
        i = route.index(u)
        return route[i + 1] if i + 1 < len(route) else None

    def on_link_fail(self, sim, u, v, pkt):
        # Route error: purge every cached route using the broken link.
        for k in list(self.cache):
            r = self.cache[k]
            for a, b in zip(r, r[1:]):
                if a == u and b == v:
                    del self.cache[k]
                    break
        pkt.route = None


# --------------------------------------------------------------------------- #
class LinkStateRouter(Router):
    """Hop-by-hop least-cost anycast routing on each node's own network view."""
    name = "etx"

    def __init__(self, spec: RouterSpec | None = None, name: str | None = None):
        if spec is not None:
            self.spec = spec
        if name:
            self.name = name
        self.uses_trust = self.spec.use_trust
        self._view_cache: dict[int, tuple] = {}

    def begin_step(self, sim):
        self._view_cache.clear()
        n = sim.n
        known = sim.known_alive
        base = sim.adv_etx.copy()
        base[~known, :] = INF
        base[:, ~known] = INF
        if self.spec.use_forecast:
            # Expected exposure: a packet handed to y waits ~(1 + queue/capacity) steps
            # there, and is lost if y becomes a ghost in the meantime.
            wait = 1.0 + sim.adv_queue / sim.cfg.node_capacity
            base = base * (1.0 + self.spec.kappa * sim.adv_risk * wait)[None, :]
        if self.spec.use_pheromone:
            base = base / (1.0 + self.spec.rho * sim.tau)
        # Gateways connect to a virtual sink (index n).
        full = np.full((n + 1, n + 1), INF)
        full[:n, :n] = base
        gw = sim.world.gateways
        full[gw, n] = 1e-3          # tiny but non-zero (scipy treats ~0 as "no edge")
        full[gw, :n] = INF          # packets are delivered on arrival at a gateway
        self._base = full

    def _view(self, sim, u: int) -> np.ndarray:
        C = self._base
        if self.spec.use_trust or sim.blacklist[u].max() > sim.t:
            C = C.copy()
        if self.spec.use_trust:
            t = sim.T[u].copy()
            t[sim.world.gateways] = 1.0
            colw = t ** (-self.spec.gamma)
            colw[t < self.spec.exclude_below] = INF
            C[:sim.n, :sim.n] *= colw[None, :]
        bl = sim.blacklist[u] > sim.t
        if bl.any():
            C[u, :sim.n][bl] = INF
        return C

    def _path(self, C, u, n):
        _, pred = dijkstra(C, directed=True, indices=u, return_predecessors=True)
        if pred[n] < 0:
            return None
        path = [n]
        while path[-1] != u:
            path.append(int(pred[path[-1]]))
        return path[::-1][:-1]          # drop the virtual sink

    def next_hop(self, sim, u, pkt):
        n = sim.n
        if u not in self._view_cache:
            C = self._view(sim, u)
            self._view_cache[u] = (C, self._path(C, u, n))
        C, path = self._view_cache[u]
        if path is not None and not (set(path[1:]) & pkt.visited):
            return path[1] if len(path) > 1 else self._evacuate(sim, u, pkt)
        # Loop prevention: exclude nodes this packet already visited.
        C2 = C.copy()
        vis = [v for v in pkt.visited if v != u]
        C2[:, vis] = INF
        path = self._path(C2, u, n)
        if path is not None and len(path) > 1:
            return path[1]
        return self._evacuate(sim, u, pkt)

    def _evacuate(self, sim, u, pkt):
        """Ghost evacuation: a node that forecasts its own death and has no route
        hands its buffered packets to the safest trusted neighbour (custody transfer)."""
        if not self.spec.use_forecast or sim.adv_risk[u] < 0.3:
            return None
        cand = sim.world.in_range[u] & sim.known_alive & (sim.blacklist[u] <= sim.t)
        cand &= sim.world.pdr[u] > 0.3
        cand[list(pkt.visited)] = False
        if self.spec.use_trust:
            cand &= sim.T[u] >= 0.5
        cand &= sim.adv_risk < 0.5 * sim.adv_risk[u]
        idx = np.where(cand)[0]
        if len(idx) == 0:
            return None
        return int(idx[np.argmin(sim.adv_risk[idx])])

    def on_delivered(self, sim, pkt):
        if self.spec.use_pheromone:
            p = pkt.path
            dep = 1.0 / max(len(p) - 1, 1)
            for a, b in zip(p, p[1:]):
                sim.tau[a, b] += dep

    def on_misbehaviour(self, sim, observer, target):
        if self.spec.use_pheromone:
            sim.tau[:, target] *= 0.5
            sim.tau[observer, target] = 0.0


# --------------------------------------------------------------------------- #
ETX_SPEC = RouterSpec(use_trust=False, use_forecast=False, use_pheromone=False,
                      asymmetric_forgetting=False, filtered_recs=False, verify_claims=False)
TRUST_ETX_SPEC = RouterSpec(use_trust=True, use_forecast=False, use_pheromone=False,
                            asymmetric_forgetting=False, filtered_recs=False, verify_claims=False)
GHOSTMESH_SPEC = RouterSpec()

ABLATIONS = {
    "ghostmesh": GHOSTMESH_SPEC,
    "−forecast": RouterSpec(use_forecast=False),
    "−pheromone": RouterSpec(use_pheromone=False),
    "−asym. evidence": RouterSpec(asymmetric_forgetting=False),
    "−rec. filtering": RouterSpec(filtered_recs=False),
    "−claim verification": RouterSpec(verify_claims=False),
    "−trust (all)": RouterSpec(use_trust=False, asymmetric_forgetting=False, filtered_recs=False,
                               verify_claims=False),
}

ROUTERS = ["flooding", "aodv", "etx", "trust_etx", "ghostmesh"]
LABELS = {"flooding": "Flooding", "aodv": "AODV", "etx": "OLSR-ETX",
          "trust_etx": "Trust-ETX", "ghostmesh": "GhostMesh"}


def make_router(name: str, spec: RouterSpec | None = None) -> Router:
    if spec is not None:
        return LinkStateRouter(spec, name)
    if name == "flooding":
        return FloodingRouter()
    if name == "aodv":
        return AODVRouter()
    if name == "etx":
        return LinkStateRouter(ETX_SPEC, "etx")
    if name == "trust_etx":
        return LinkStateRouter(TRUST_ETX_SPEC, "trust_etx")
    if name == "ghostmesh":
        return LinkStateRouter(GHOSTMESH_SPEC, "ghostmesh")
    raise ValueError(f"unknown router {name!r}")
