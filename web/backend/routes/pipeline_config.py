"""Endpoints for pipeline configuration."""

from fastapi import APIRouter

from ..config import cfg

router = APIRouter()


@router.get("")
async def get_config():
    """Return the current pipeline config.yaml parsed."""
    return cfg.pipeline_config


@router.get("/training")
async def get_training_config():
    """Return just the training section."""
    return cfg.pipeline_config.get("training", {})
