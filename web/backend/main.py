"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import cfg
from .routes import runs, evals, diagnostics, pipeline_config, cycle, models, agent
from .ws import router as ws_router

app = FastAPI(title="trainLLM Dashboard", version="0.1.0")

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
