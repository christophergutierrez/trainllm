"""Endpoints for evaluation results exploration."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ..config import cfg
from .. import charts

router = APIRouter()


@router.get("/{eval_id}")
async def get_eval(eval_id: str):
    """Get full eval results including all records."""
    path = cfg.evals_dir / f"{eval_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Eval not found: {eval_id}")
    data = json.loads(path.read_text())
    return data


@router.get("/{eval_id}/records")
async def get_eval_records(
    eval_id: str,
    band: str | None = Query(None),
    convention: str | None = Query(None),
    min_score: float | None = Query(None),
    max_score: float | None = Query(None),
    sort_by: str = Query("composite_score"),
    sort_desc: bool = Query(True),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """Get paginated, filtered eval records."""
    path = cfg.evals_dir / f"{eval_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Eval not found: {eval_id}")

    data = json.loads(path.read_text())
    records = data.get("results", [])

    if band:
        records = [r for r in records if r.get("band") == band.upper()]
    if convention:
        records = [r for r in records if convention in r.get("conventions", [])]
    if min_score is not None:
        records = [r for r in records if r.get("composite_score", r.get("score", 0)) >= min_score]
    if max_score is not None:
        records = [r for r in records if r.get("composite_score", r.get("score", 0)) <= max_score]

    def sort_key(r):
        return r.get(sort_by, r.get("score", 0))

    records.sort(key=sort_key, reverse=sort_desc)

    return {
        "total": len(records),
        "offset": offset,
        "limit": limit,
        "records": records[offset:offset + limit],
    }


@router.get("/{eval_id}/charts/bands")
async def eval_band_chart(eval_id: str):
    """Get band distribution donut chart."""
    path = cfg.evals_dir / f"{eval_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Eval not found: {eval_id}")
    data = json.loads(path.read_text())
    band_counts = data.get("summary", {}).get("band_counts", {})
    if not band_counts:
        raise HTTPException(404, "No band data")
    return charts.band_donut(band_counts)


@router.get("/{eval_id}/charts/scores")
async def eval_score_distribution(eval_id: str):
    """Get score distribution histogram."""
    path = cfg.evals_dir / f"{eval_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Eval not found: {eval_id}")
    data = json.loads(path.read_text())
    scores = [
        r.get("composite_score", r.get("score", 0))
        for r in data.get("results", [])
        if r.get("band") != "ERROR"
    ]
    if not scores:
        return charts.score_distribution([0.0])
    return charts.score_distribution(scores)
