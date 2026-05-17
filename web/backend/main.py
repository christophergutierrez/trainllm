"""FastAPI application entry point."""

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import cfg
from .routes import runs, evals, diagnostics, pipeline_config, cycle, models, agent
from .ws import router as ws_router, manager, Channel

async def _tail_events():
    """Background task: tail the training events file and broadcast new lines."""
    path = cfg.events_file
    pos = 0
    if path.exists():
        pos = path.stat().st_size

    while True:
        await asyncio.sleep(1)
        if not path.exists():
            pos = 0
            continue
        size = path.stat().st_size
        if size < pos:
            pos = 0
        if size == pos:
            continue
        try:
            with open(path) as f:
                f.seek(pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        await manager.broadcast(Channel.TRAINING, event)
                    except json.JSONDecodeError:
                        pass
                pos = f.tell()
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_tail_events())
    yield
    task.cancel()


app = FastAPI(title="trainLLM Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(evals.router, prefix="/api/evals", tags=["evals"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])
app.include_router(pipeline_config.router, prefix="/api/config", tags=["config"])
app.include_router(cycle.router, prefix="/api/cycle", tags=["cycle"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "adapter": cfg.adapter_name}
