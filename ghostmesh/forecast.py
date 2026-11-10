"""Ghost forecaster: predicts which nodes will fail within the next H steps.

Learned *on the network, without a server*:

* Each node runs the same logistic-regression model on telemetry it already
  broadcasts in HELLO beacons (battery level and slope, a temperature-like
  stress reading and its slope, recent neighbour churn, mean link quality).
* Labels arrive with a delay: a snapshot taken at time t is labelled once the
  horizon t + H has passed (did the node die in between?). A node cannot learn
  from its own death, so the *neighbours* that heard its last beacons train on it.
* After each local SGD step, nodes average weights with their neighbours
  (gossip learning / decentralised federated averaging), so knowledge about
  how nodes die spreads through the mesh.

Each node then advertises its own failure probability in its link-state
updates; GhostMesh's edge cost penalises next hops that are likely to become
ghosts before the packet gets through.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FEATURES = ["bias", "battery", "battery_urgency", "stress", "stress_slope", "neigh_churn", "link_quality"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GhostForecaster:
    def __init__(self, n: int, horizon: int, lr: float = 0.15, pos_weight: float = 6.0, gossip: bool = True,
                 rng: np.random.Generator | None = None):
        self.n = n
        self.h = horizon
        self.lr = lr
        self.pos_weight = pos_weight
        self.gossip = gossip
        self.d = len(FEATURES)
        self.W = np.zeros((n, self.d))          # one model per node
        self.W[:, 0] = -2.0                      # prior: failures are rare
        self.hist_batt = deque(maxlen=6)
        self.hist_stress = deque(maxlen=6)
        self.hist_alive = deque(maxlen=11)
        # (t, X, alive_mask, neighbour_matrix) awaiting labels
        self.pending: deque = deque()
        self.rng = rng or np.random.default_rng()
        # Evaluation log: (prob, label, battery-only baseline score)
        self.eval_p, self.eval_y, self.eval_base = [], [], []
        self.eval_X, self.eval_t = [], []

    def features(self, battery, stress_obs, alive, in_range, pdr) -> np.ndarray:
        n = self.n
        b = np.where(np.isfinite(battery), battery, 1.0)
        self.hist_batt.append(b.copy())
        self.hist_stress.append(stress_obs.copy())
        self.hist_alive.append(alive.copy())
        steps = max(len(self.hist_batt) - 1, 1)
        drain = (self.hist_batt[0] - self.hist_batt[-1]) / steps if len(self.hist_batt) > 1 else np.zeros(n)
        # Time-to-empty at the current drain rate, squashed to (0, 1]: 1 = empty within ~H steps.
        tte = b / np.maximum(drain, 1e-5)
        urgency = np.exp(-tte / self.h)
        s_slope = (self.hist_stress[-1] - self.hist_stress[0]) * 5 if len(self.hist_stress) > 1 else np.zeros(n)
        lost = self.hist_alive[0] & ~alive
        deg0 = (in_range & self.hist_alive[0][None, :]).sum(1)
        churn = (in_range & lost[None, :]).sum(1) / np.maximum(deg0, 1)
        live_links = pdr * alive[None, :]
        lq = live_links.sum(1) / np.maximum((live_links > 0).sum(1), 1)
        X = np.stack([np.ones(n), b, urgency, stress_obs, s_slope, churn, lq], 1)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(np.einsum("nd,nd->n", self.W, X))

    def step(self, t: int, X: np.ndarray, alive: np.ndarray, neighbours: np.ndarray,
             death_time: np.ndarray, eligible: np.ndarray):
        """Record today's snapshot, train on snapshots whose horizon has elapsed, gossip."""
        p = self.predict(X)
        self.pending.append((t, X.copy(), (alive & eligible).copy(), neighbours.copy(), p.copy()))
        while self.pending and self.pending[0][0] <= t - self.h:
            t0, X0, mask0, nb0, p0 = self.pending.popleft()
            y = ((death_time > t0) & (death_time <= t0 + self.h)).astype(float)
            idx = np.where(mask0)[0]
            if len(idx) == 0:
                continue
            self.eval_p.extend(p0[idx].tolist())
            self.eval_y.extend(y[idx].tolist())
            self.eval_base.extend((1 - X0[idx, 1]).tolist())
            self.eval_X.append(X0[idx])
            self.eval_t.extend([t0] * len(idx))
            # Learners for snapshot k: k itself (if still alive) and anyone who heard it.
            learn = nb0[:, idx] | np.diag(np.ones(self.n, bool))[:, idx]   # (learner, sample)
            learn &= alive[:, None]
            Xs, ys = X0[idx], y[idx]
            P = _sigmoid(self.W @ Xs.T)                                  # (learner, sample)
            cw = np.where(ys > 0, self.pos_weight, 1.0)
            G = ((P - ys[None, :]) * cw[None, :] * learn) @ Xs           # (learner, d)
            cnt = learn.sum(1, keepdims=True)
            self.W -= self.lr * G / np.maximum(cnt, 1) + 1e-4 * self.W * (cnt > 0)
        if not self.gossip:
            return p
        # Gossip averaging with neighbours (row-stochastic mixing among live nodes).
        A = (neighbours & alive[None, :] & alive[:, None]).astype(float)
        np.fill_diagonal(A, 1.0)
        A /= A.sum(1, keepdims=True)
        self.W = A @ self.W
        return p
