"""Scenario and protocol configuration.

Every experiment in this repository is fully described by a :class:`Config`
plus a router name, so results are reproducible from a seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Tuple


@dataclass(frozen=True)
class Config:
    # ---------------------------------------------------------------- topology
    n_nodes: int = 50            # total nodes, gateways included
    n_gateways: int = 3          # uplink nodes (drone / satellite relays)
    radio_range: float = 0.24    # in units of the (1 x 1) deployment area
    steps: int = 300
    seed: int = 0

    # ----------------------------------------------------------------- traffic
    traffic_rate: float = 1.5    # mean new SOS packets per step (Poisson)
    ttl: int = 40                # packet lifetime in steps (store-carry-forward)
    node_capacity: int = 3       # transmissions per node per step
    max_retries: int = 3         # link-layer retries before a link is declared broken

    # ------------------------------------------------------------------ energy
    battery_range: Tuple[float, float] = (0.45, 1.0)
    e_tx: float = 0.004
    e_rx: float = 0.0010
    e_idle: float = 0.0007       # includes periodic HELLO / link-state beacons
    e_ctrl: float = 0.0003       # per control transmission (route discovery floods)

    # ---------------------------------------------------------------- disaster
    front_start: int = 30        # step at which the hazard front ignites
    front_speed: float = 0.0012  # radius growth per step
    hazard_rate: float = 0.035   # per-step death probability inside the front
    base_failure: float = 0.0004 # unpredictable per-step hardware failure
    detect_delay: int = 6        # HELLO timeout: steps before a dead node is noticed

    # --------------------------------------------------------------- adversary
    malicious_frac: float = 0.20
    attack_start: int = 20
    attack_mix: Tuple[str, ...] = ("blackhole", "grayhole", "onoff")
    grayhole_drop: float = 0.5
    onoff_period: int = 40       # on for half the period, off for the other half
    lie_links: bool = True       # attackers advertise perfect links (route attraction)
    badmouth: bool = True        # attackers send false trust recommendations

    # ------------------------------------------------------------------- trust
    trust_prior: Tuple[float, float] = (2.0, 1.0)  # Beta(alpha, beta) prior
    rec_period: int = 5          # recommendation exchange period (steps)
    rec_discount: float = 0.35   # weight of second-hand evidence
    watch_deadline: int = 4      # steps a watchdog waits to overhear a forward

    # -------------------------------------------------------------- forecaster
    horizon: int = 12            # predict "dies within H steps"
    fc_lr: float = 1.5
    fc_pos_weight: float = 2.0
    fc_gossip: bool = True       # average model weights with neighbours (decentralised learning)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


@dataclass(frozen=True)
class RouterSpec:
    """Knobs of the trust-aware routers. GhostMesh ablations toggle these."""
    use_trust: bool = True
    asymmetric_forgetting: bool = True   # remember misbehaviour longer than good behaviour
    filtered_recs: bool = True           # reject outlier / untrusted recommendations
    verify_claims: bool = True           # cross-check advertised link quality against own measurement
    use_forecast: bool = True            # penalise nodes predicted to fail soon
    use_pheromone: bool = True           # stigmergic reinforcement of delivering paths
    gamma: float = 2.0                   # trust exponent in the edge cost
    kappa: float = 4.0                   # failure-risk penalty
    rho: float = 0.6                     # pheromone attraction
    exclude_below: float = 0.35          # nodes with trust below this are never used
    extra: dict = field(default_factory=dict)
