"""GhostMesh: predictive, trust-aware routing for emergency mesh networks."""
from .config import Config, RouterSpec
from .routers import ROUTERS, LABELS, ABLATIONS, make_router
from .sim import Simulation

__all__ = ["Config", "RouterSpec", "Simulation", "ROUTERS", "LABELS", "ABLATIONS", "make_router"]
__version__ = "1.0.0"
