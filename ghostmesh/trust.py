"""Decentralised Bayesian trust.

Every node ``i`` keeps a Beta(alpha, beta) belief about every other node ``k``:

* **Direct evidence** comes from a watchdog. When ``i`` hands a packet to ``k``
  it listens for ``k`` re-transmitting it. Overheard -> alpha += 1, silence until
  the deadline -> beta += 1. The watchdog is noisy (missed overhearing, honest
  nodes that are merely buffering), so honest nodes collect some false negatives.
* **Indirect evidence** comes from neighbours' recommendations every
  ``rec_period`` steps. Attackers *badmouth* (claim honest nodes are bad and
  colluders are good).

Two GhostMesh defences, each switchable for ablation:

1. **Asymmetric evidence**: a dropped packet weighs twice a forwarded one and
   bad evidence decays far slower than good evidence, so grey-holes cannot hide
   behind their partial forwarding and on-off attackers cannot "wash" their
   reputation during an off phase.
2. **Filtered recommendations**: second-hand opinions are accepted only from
   neighbours ``i`` already trusts, and are discarded when they deviate strongly
   from ``i``'s own first-hand opinion (a majority-rule / deviation test).
"""
from __future__ import annotations

import numpy as np


class TrustModel:
    def __init__(self, n: int, prior=(2.0, 1.0), asymmetric=True, filtered=True,
                 rec_discount=0.35, rng: np.random.Generator | None = None):
        self.n = n
        self.a0, self.b0 = prior
        self.alpha = np.full((n, n), self.a0)
        self.beta = np.full((n, n), self.b0)
        self.alpha_r = np.zeros((n, n))   # indirect evidence (recomputed each round)
        self.beta_r = np.zeros((n, n))
        self.cred_a = np.ones((n, n))     # recommendation credibility (filtered mode)
        self.cred_b = np.ones((n, n))
        self.asymmetric = asymmetric
        self.filtered = filtered
        self.rec_discount = rec_discount
        # Forgetting factors applied to evidence above the prior.
        self.lam_pos = 0.985
        self.lam_neg = 0.998 if asymmetric else 0.985
        # "Hard to earn, easy to lose": a drop weighs more than a forward.
        self.neg_weight = 1.5 if asymmetric else 1.0
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------ evidence
    def observe(self, i: int, k: int, good: bool, weight: float = 1.0):
        if good:
            self.alpha[i, k] += weight
        else:
            self.beta[i, k] += weight * self.neg_weight

    def decay(self):
        self.alpha = self.a0 + (self.alpha - self.a0) * self.lam_pos
        self.beta = self.b0 + (self.beta - self.b0) * self.lam_neg

    # -------------------------------------------------------------- beliefs
    def direct(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def trust(self) -> np.ndarray:
        """Combined trust matrix T[i, k] = i's belief that k forwards honestly."""
        a = self.alpha + self.alpha_r
        b = self.beta + self.beta_r
        t = a / (a + b)
        np.fill_diagonal(t, 1.0)
        return t

    # ------------------------------------------------------ recommendations
    def exchange(self, neighbours: np.ndarray, malicious: np.ndarray, badmouth: bool):
        """One round of recommendation gossip.

        ``neighbours[i, j]`` is True when i can currently hear j. Malicious
        recommenders badmouth honest nodes and praise colluders when ``badmouth``.

        Filtered mode keeps a separate *recommendation credibility* Beta(c_a, c_b)
        per (i, j): whenever j recommends about a node k that i has first-hand
        experience with, agreement raises and disagreement lowers i's credibility
        in j. Only credible recommenders are listened to, and a recommendation that
        contradicts i's own first-hand opinion is discarded.
        """
        n = self.n
        cap = 5.0                                # max evidence any single recommendation carries
        ev_a = self.alpha - self.a0
        ev_b = self.beta - self.b0
        shared_a, shared_b = ev_a.copy(), ev_b.copy()
        if badmouth and malicious.any():
            liars = np.where(malicious)[0]
            shared_a[liars] = np.where(malicious, cap, 0.0)[None, :]
            shared_b[liars] = np.where(malicious, 0.0, cap)[None, :]
        tot = shared_a + shared_b
        scale = np.minimum(1.0, cap / np.maximum(tot, 1e-9))
        shared_a, shared_b = shared_a * scale, shared_b * scale
        shared_ev = shared_a + shared_b
        shared_e = (self.a0 + shared_a) / (self.a0 + self.b0 + shared_ev)

        t_dir = self.direct()
        own_ev = ev_a + ev_b
        nb = neighbours.copy()
        np.fill_diagonal(nb, False)
        if self.filtered:
            # Update recommendation credibility from overlap with first-hand experience.
            both = (own_ev[:, None, :] >= 2.0) & (shared_ev[None, :, :] >= 1.0) & nb[:, :, None]
            dev = np.abs(shared_e[None, :, :] - t_dir[:, None, :])          # (i, j, k)
            agree = (both & (dev < 0.3)).sum(2)
            disagree = (both & (dev >= 0.3)).sum(2)
            self.cred_a = 1.0 + (self.cred_a - 1.0) * 0.97 + 0.5 * agree
            self.cred_b = 1.0 + (self.cred_b - 1.0) * 0.97 + 0.5 * disagree
            cred = self.cred_a / (self.cred_a + self.cred_b)
            w = nb * (cred >= 0.5) * cred ** 2 * (t_dir >= 0.5)
            accept = np.broadcast_to(w[:, :, None] > 0, (n, n, n)).copy()
            accept &= ~((own_ev[:, None, :] >= 3.0) & (dev > 0.4))
        else:
            w = nb * t_dir
            accept = np.broadcast_to(w[:, :, None] > 0, (n, n, n)).copy()
        wk = w[:, :, None] * accept                      # (i, j, k)
        norm = wk.sum(1)                                 # (i, k)
        has = norm > 0
        norm = np.where(has, norm, 1.0)
        self.alpha_r = self.rec_discount * np.einsum("ijk,jk->ik", wk, shared_a) / norm * has
        self.beta_r = self.rec_discount * np.einsum("ijk,jk->ik", wk, shared_b) / norm * has
        np.fill_diagonal(self.alpha_r, 0.0)
        np.fill_diagonal(self.beta_r, 0.0)

    def credibility(self) -> np.ndarray:
        return self.cred_a / (self.cred_a + self.cred_b)
