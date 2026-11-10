"""Physical world: node placement, radio links, energy, and the disaster front.

The world knows the *truth* (who is alive, real link quality). Routers only
see what the protocol would see: link-state advertisements, which can be stale
(dead nodes linger for ``detect_delay`` steps as *ghosts*) or forged by attackers.
"""
from __future__ import annotations

import numpy as np

from .config import Config


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class World:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n = cfg.n_nodes
        self.n = n

        self.gateways = np.arange(cfg.n_gateways)
        self.is_gateway = np.zeros(n, bool)
        self.is_gateway[self.gateways] = True

        self.pos = self._place_nodes()
        d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        self.dist = d
        self.in_range = (d < cfg.radio_range) & ~np.eye(n, dtype=bool)
        # Static edge list (for compact traces); quality varies over time.
        iu, ju = np.where(np.triu(self.in_range))
        self.edges = np.stack([iu, ju], 1)

        # Mean link quality from distance, plus AR(1) shadow fading in logit space.
        self._pdr_logit_mean = -(d / cfg.radio_range - 0.72) / 0.075
        self._fade = np.zeros((n, n))

        lo, hi = cfg.battery_range
        self.battery = rng.uniform(lo, hi, n)
        self.battery[self.is_gateway] = np.inf
        self.alive = np.ones(n, bool)
        self.death_time = np.full(n, np.inf)
        self.death_cause = np.array([""] * n, dtype=object)

        # Hazard front (e.g. wildfire / flood) expanding from an epicentre.
        self.front_origin = self._pick_origin()
        self.front_radius = 0.0

        self.pdr = np.zeros((n, n))
        self.update_links()

    # ------------------------------------------------------------- placement
    def _place_nodes(self) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        n, g = cfg.n_nodes, cfg.n_gateways
        gw_anchor = np.array([[0.06, 0.06], [0.94, 0.10], [0.50, 0.95], [0.06, 0.90], [0.94, 0.92]])
        for _ in range(500):
            pos = rng.uniform(0.02, 0.98, (n, 2))
            pos[:g] = gw_anchor[:g] + rng.normal(0, 0.015, (g, 2))
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            adj = (d < cfg.radio_range * 0.85) & ~np.eye(n, dtype=bool)
            if _connected(adj):
                return pos
        raise RuntimeError("could not place a connected topology; increase radio_range")

    def _pick_origin(self) -> np.ndarray:
        for _ in range(200):
            o = self.rng.uniform(0.3, 0.7, 2)
            if np.min(np.linalg.norm(self.pos[self.gateways] - o, axis=1)) > 0.35:
                return o
        return np.array([0.5, 0.5])

    # ---------------------------------------------------------------- dynamics
    def update_links(self):
        n = self.n
        noise = self.rng.normal(0, 1.0, (n, n))
        noise = (noise + noise.T) / np.sqrt(2)
        # Stationary std of the fade term is 0.7 logits.
        self._fade = 0.9 * self._fade + np.sqrt(1 - 0.81) * 0.7 * noise
        logit = self._pdr_logit_mean + self._fade
        # Links touching nodes inside / near the front degrade (smoke, debris, heat).
        stress = self.stress()
        degrade = 1.0 - 0.45 * np.clip(np.maximum(stress[:, None], stress[None, :]), 0, 1)
        pdr = _logistic(logit) * degrade
        pdr[~self.in_range] = 0.0
        np.fill_diagonal(pdr, 0.0)
        self.pdr = pdr

    def dist_to_front(self) -> np.ndarray:
        """Signed distance: negative = inside the front."""
        if self.front_radius <= 0:
            return np.full(self.n, 1.0)
        return np.linalg.norm(self.pos - self.front_origin, axis=1) - self.front_radius

    def stress(self) -> np.ndarray:
        """Environmental stress in [0, ~1.5]; the quantity a temperature sensor would see."""
        dtf = self.dist_to_front()
        return np.where(dtf < 0, 1.0 + np.minimum(-dtf, 0.1) * 5, np.exp(-dtf / 0.07))

    def step_hazards(self, t: int, tx_count: np.ndarray, rx_count: np.ndarray):
        """Advance the front, drain batteries, kill nodes. Returns newly dead node ids."""
        cfg, rng = self.cfg, self.rng
        if t >= cfg.front_start:
            self.front_radius += cfg.front_speed

        alive = self.alive & ~self.is_gateway
        drain = cfg.e_idle + cfg.e_tx * tx_count + cfg.e_rx * rx_count
        self.battery[alive] -= drain[alive]

        dtf = self.dist_to_front()
        inside = dtf < 0
        p_die = np.full(self.n, cfg.base_failure)
        p_die[inside] += cfg.hazard_rate * (1 + np.minimum(-dtf[inside], 0.15) * 10)
        dies = alive & ((self.battery <= 0) | (rng.random(self.n) < p_die))
        new_dead = np.where(dies)[0]
        for i in new_dead:
            self.alive[i] = False
            self.death_time[i] = t
            self.death_cause[i] = "battery" if self.battery[i] <= 0 else ("hazard" if inside[i] else "hardware")
        return new_dead

    def known_alive(self, t: int) -> np.ndarray:
        """What the routing layer believes: dead nodes stay 'alive' until the HELLO timeout."""
        return self.alive | (t - self.death_time < self.cfg.detect_delay)


def _connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    seen = np.zeros(n, bool)
    seen[0] = True
    frontier = [0]
    while frontier:
        u = frontier.pop()
        for v in np.where(adj[u] & ~seen)[0]:
            seen[v] = True
            frontier.append(v)
    return bool(seen.all())
