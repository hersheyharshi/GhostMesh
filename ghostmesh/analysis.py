"""Graph-theoretic health of the mesh (NetworkX).

Answers questions the routing layer cannot: is the network physically able to
deliver at all, which relays are single points of failure, and how close is
any protocol to the best achievable delivery?
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def mesh_graph(sim, honest_only: bool = False, min_quality: float = 0.1) -> nx.Graph:
    w = sim.world
    keep = w.alive.copy()
    if honest_only:
        keep &= ~sim.malicious
    g = nx.Graph()
    for i in np.where(keep)[0]:
        g.add_node(int(i), gateway=bool(w.is_gateway[i]), malicious=bool(sim.malicious[i]))
    for a, b in w.edges:
        if keep[a] and keep[b] and w.pdr[a, b] >= min_quality:
            g.add_edge(int(a), int(b), etx=1.0 / w.pdr[a, b])
    return g


def _reach_fraction(g: nx.Graph, gateways, sources) -> float:
    if not sources:
        return float("nan")
    reach = set()
    for gw in gateways:
        if gw in g:
            reach |= nx.node_connected_component(g, gw)
    return sum(1 for s in sources if s in reach) / len(sources)


def topology_report(sim) -> dict:
    """Snapshot metrics for the current step of a :class:`~ghostmesh.Simulation`."""
    w = sim.world
    gws = [int(x) for x in w.gateways]
    honest_relays = [int(i) for i in np.where(w.alive & ~w.is_gateway & ~sim.malicious)[0]]
    g_all = mesh_graph(sim)
    g_hon = mesh_graph(sim, honest_only=True)
    giant = max(nx.connected_components(g_all), key=len) if len(g_all) else set()
    sub = g_all.subgraph(giant)
    lam2 = float(nx.algebraic_connectivity(sub, weight=None)) if len(sub) > 2 else 0.0
    cut = sorted(int(v) for v in nx.articulation_points(g_all) if not w.is_gateway[v])
    return {
        "t": sim.t,
        "alive_nodes": int(w.alive.sum()),
        "components": nx.number_connected_components(g_all),
        "algebraic_connectivity": round(lam2, 4),
        "articulation_points": cut,
        # Best case for any protocol: fraction of honest relays with *some* path to a gateway.
        "reachable_any_path": _reach_fraction(g_all, gws, honest_relays),
        # Best case if every attacker were perfectly avoided.
        "reachable_honest_path": _reach_fraction(g_hon, gws, honest_relays),
    }
