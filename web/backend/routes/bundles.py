"""Endpoints for bundle management and download."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import cfg

router = APIRouter()

BUNDLE_DIR = cfg.base_dir / "bundles"
REGISTRY_PATH = cfg.base_dir / "bundles.json"


@router.get("")
async def list_bundles():
    """List all bundled artifacts from the registry."""
    if not REGISTRY_PATH.exists():
        return {"bundles": []}
    data = json.loads(REGISTRY_PATH.read_text())
    return data


@router.get("/{bundle_name}")
async def get_bundle_manifest(bundle_name: str):
    """Get the manifest from a specific bundle without downloading it."""
    import tarfile

    tarball = BUNDLE_DIR / f"{bundle_name}.tar.gz"
    if not tarball.exists():
        raise HTTPException(404, f"Bundle not found: {bundle_name}")

    with tarfile.open(tarball, "r:gz") as tar:
        manifest_member = f"{bundle_name}/manifest.json"
        try:
            f = tar.extractfile(manifest_member)
            if f is None:
                raise HTTPException(500, "Manifest not readable")
            return json.loads(f.read())
        except KeyError:
            raise HTTPException(500, "Bundle missing manifest.json")


@router.get("/{bundle_name}/download")
async def download_bundle(bundle_name: str):
    """Download a bundle tarball."""
    tarball = BUNDLE_DIR / f"{bundle_name}.tar.gz"
    if not tarball.exists():
        raise HTTPException(404, f"Bundle not found: {bundle_name}")
    return FileResponse(
        path=str(tarball),
        media_type="application/gzip",
        filename=f"{bundle_name}.tar.gz",
    )
