"""FastAPI + WebSocket backend for the live dashboard.

    pip install -r requirements.txt
    uvicorn server.app:app --reload
    open http://localhost:8000          # dashboard switches to LIVE mode automatically

Protocol (JSON over ``/ws``)
----------------------------
client -> server  {"cmd": "start", "seed": 3, "baseline": "etx", "malicious_frac": 0.2}
                  {"cmd": "pause"} | {"cmd": "resume"} | {"cmd": "speed", "fps": 12}
                  {"cmd": "kill", "node": 17} | {"cmd": "compromise", "node": 17}
server -> client  {"type": "static", ...}  once per start
                  {"type": "frame", "frames": {"baseline": {...}, "ghostmesh": {...}}}
                  {"type": "done"}
"""
from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ghostmesh import Config, ROUTERS, Simulation
from .live import LiveSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")

app = FastAPI(title="GhostMesh", version="1.0.0",
              description="Predictive, trust-aware routing for emergency mesh networks")


@app.get("/")
def index():
    return FileResponse(os.path.join(DOCS, "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "routers": ROUTERS}


@app.get("/api/run")
def run(router: str = "ghostmesh", seed: int = 0, malicious_frac: float = 0.2, steps: int = 300):
    """Run one headless simulation and return its summary metrics."""
    if router not in ROUTERS:
        return JSONResponse({"error": f"router must be one of {ROUTERS}"}, status_code=400)
    s = Simulation(Config(seed=seed, malicious_frac=malicious_frac, steps=min(steps, 600)), router).run()
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in s.items()}


@app.get("/api/topology")
def topology(seed: int = 0, step: int = 150, malicious_frac: float = 0.2):
    """Graph-theoretic health of the mesh at a given step (NetworkX)."""
    from ghostmesh.analysis import topology_report
    sim = Simulation(Config(seed=seed, malicious_frac=malicious_frac), "etx")
    for _ in range(max(0, min(step, 600))):
        sim.step()
    return topology_report(sim)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    session: LiveSession | None = None
    paused, fps = False, 10.0
    inbox: asyncio.Queue = asyncio.Queue()

    async def reader():
        try:
            while True:
                await inbox.put(await websocket.receive_json())
        except WebSocketDisconnect:
            await inbox.put({"cmd": "_closed"})

    task = asyncio.create_task(reader())
    try:
        while True:
            while not inbox.empty() or session is None or paused:
                msg = await inbox.get()
                cmd = msg.get("cmd")
                if cmd == "_closed":
                    return
                if cmd == "start":
                    try:
                        session = LiveSession(int(msg.get("seed", 3)), msg.get("baseline", "etx"),
                                              float(msg.get("malicious_frac", 0.2)))
                    except ValueError as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                        continue
                    paused = False
                    await websocket.send_json(session.static())
                elif cmd == "pause":
                    paused = True
                elif cmd == "resume":
                    paused = False
                elif cmd == "speed":
                    fps = max(1.0, min(60.0, float(msg.get("fps", 10))))
                elif session is not None:
                    session.command(msg)
                if inbox.empty() and session is not None and not paused:
                    break
            if session.done:
                await websocket.send_json({"type": "done"})
                paused = True
                continue
            frame = await asyncio.to_thread(session.step)
            await websocket.send_json(frame)
            await asyncio.sleep(1.0 / fps)
    finally:
        task.cancel()


app.mount("/", StaticFiles(directory=DOCS, html=True), name="docs")
