"""Framework-free live session: two simulations of the *same* disaster in lockstep.

Kept separate from the FastAPI app so it can be unit-tested without a server.
"""
from __future__ import annotations

from ghostmesh import Config, Simulation
from ghostmesh.routers import ROUTERS


class LiveSession:
    def __init__(self, seed: int = 3, baseline: str = "etx", malicious_frac: float = 0.2,
                 steps: int = 300):
        if baseline not in ROUTERS or baseline == "ghostmesh":
            raise ValueError(f"baseline must be one of {[r for r in ROUTERS if r != 'ghostmesh']}")
        cfg = Config(seed=seed, malicious_frac=malicious_frac, steps=steps)
        self.cfg = cfg
        self.sims = {"baseline": Simulation(cfg, baseline), "ghostmesh": Simulation(cfg, "ghostmesh")}
        self.baseline = baseline

    @property
    def done(self) -> bool:
        return self.sims["ghostmesh"].t >= self.cfg.steps

    def static(self) -> dict:
        st = self.sims["ghostmesh"].static_frame()
        st["routers"] = {"baseline": self.baseline, "ghostmesh": "ghostmesh"}
        return {"type": "static", **st}

    def step(self) -> dict:
        frames = {}
        for k, sim in self.sims.items():
            sim.step()
            frames[k] = sim.frame()
            frames[k]["t"] = sim.t - 1
        return {"type": "frame", "frames": frames}

    def command(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        node = msg.get("node")
        if cmd == "kill" and isinstance(node, int):
            for sim in self.sims.values():
                sim.kill(node)
        elif cmd == "compromise" and isinstance(node, int):
            for sim in self.sims.values():
                sim.compromise(node, msg.get("kind", "blackhole"))
