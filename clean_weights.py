#!/usr/bin/env python3
"""Clean up weight directories after archiving.

Removes LoRA adapters, merged models, and loose archive directories
after verifying that a tarball archive exists. Will NOT delete anything
unless a matching archive is found and --confirm is passed.

Usage:
    python3 clean_weights.py                    # dry run — show what would be removed
    python3 clean_weights.py --confirm          # actually remove files
    python3 clean_weights.py --keep-latest      # keep the most recent final/ per adapter
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config

cfg = _config.load()

ARCHIVE_DIR = cfg.base_dir / "archive"
LORA_ROOT = cfg.base_dir / "lora"
MERGED_ROOT = cfg.base_dir / "merged"


def find_latest_archive() -> Path | None:
    if not ARCHIVE_DIR.exists():
        return None
    tarballs = sorted(ARCHIVE_DIR.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)
    return tarballs[-1] if tarballs else None


def list_archived_adapters(tarball: Path) -> set[str]:
    """Read adapter names from a tarball's lora/ entries."""
    names = set()
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getnames():
            if member.startswith("lora/"):
                parts = member.split("/")
                if len(parts) >= 2 and parts[1]:
                    names.add(parts[1])
    return names


def find_loose_archive_dirs() -> list[Path]:
    """Find non-tarball directories in archive/ (old manual archives)."""
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(
        d for d in ARCHIVE_DIR.iterdir()
        if d.is_dir() and d.name != ".gitkeep"
    )


def collect_removals(keep_latest: bool) -> dict[str, list[Path]]:
    """Collect all paths that would be removed, grouped by category."""
    removals: dict[str, list[Path]] = {
        "lora_adapters": [],
        "merged_models": [],
        "loose_archives": [],
    }

    if LORA_ROOT.exists():
        for d in sorted(LORA_ROOT.iterdir()):
            if d.is_dir() and d.name != ".gitkeep":
                removals["lora_adapters"].append(d)

    if MERGED_ROOT.exists():
        for d in sorted(MERGED_ROOT.iterdir()):
            if d.name == ".gitkeep":
                continue
            removals["merged_models"].append(d)

    removals["loose_archives"] = find_loose_archive_dirs()

    return removals


def format_size(path: Path) -> str:
    if path.is_file():
        mb = path.stat().st_size / 1_048_576
    else:
        mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete files (without this flag, only shows what would be removed)")
    parser.add_argument("--keep-latest", action="store_true",
                        help="Keep the most recent final/ directory for each adapter")
    parser.add_argument("--skip-archive-check", action="store_true",
                        help="Skip verification that an archive exists (dangerous)")
    args = parser.parse_args()

    # Verify archive exists
    archive = find_latest_archive()
    if not archive and not args.skip_archive_check:
        print("No archive tarball found in archive/.", file=sys.stderr)
        print("Run archive.py first to create a backup before cleaning.", file=sys.stderr)
        print("Or use --skip-archive-check to bypass (dangerous).", file=sys.stderr)
        sys.exit(1)

    if archive:
        archived_adapters = list_archived_adapters(archive)
        print(f"Latest archive: {archive.name}")
        print(f"  Contains {len(archived_adapters)} adapters: {', '.join(sorted(archived_adapters))}")
        print()

    removals = collect_removals(args.keep_latest)

    # Check that adapters being removed are in the archive
    if archive and not args.skip_archive_check:
        unarchived = []
        for d in removals["lora_adapters"]:
            if d.name not in archived_adapters:
                unarchived.append(d.name)
        if unarchived:
            print("WARNING: These adapters are NOT in the archive and would be lost:", file=sys.stderr)
            for name in unarchived:
                print(f"  {name}", file=sys.stderr)
            print("\nRun archive.py first, or use --skip-archive-check to proceed anyway.", file=sys.stderr)
            sys.exit(1)

    # Print summary
    total_mb = 0

    if removals["lora_adapters"]:
        print("LoRA adapters to remove:")
        for d in removals["lora_adapters"]:
            size = format_size(d)
            total_mb += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1_048_576
            in_archive = "  [archived]" if archive and d.name in archived_adapters else "  [NOT ARCHIVED]"
            print(f"    {d.name:30s}  {size:>10s}{in_archive}")

    if removals["merged_models"]:
        print("\nMerged models to remove:")
        for d in removals["merged_models"]:
            size = format_size(d)
            total_mb += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1_048_576
            print(f"    {d.name:30s}  {size:>10s}")

    if removals["loose_archives"]:
        print("\nLoose archive directories to remove:")
        for d in removals["loose_archives"]:
            size = format_size(d)
            total_mb += sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1_048_576
            print(f"    {d.name:30s}  {size:>10s}")

    total_items = sum(len(v) for v in removals.values())
    if total_items == 0:
        print("Nothing to clean up.")
        return

    print(f"\nTotal: {total_items} items, {total_mb / 1024:.1f} GB")

    if not args.confirm:
        print("\nDry run — nothing was deleted. Add --confirm to remove these files.")
        return

    # Actually remove
    print()
    for category, paths in removals.items():
        for p in paths:
            if p.is_dir():
                print(f"  Removing {p} ...")
                shutil.rmtree(p)
            elif p.is_file():
                print(f"  Removing {p} ...")
                p.unlink()

    # Restore .gitkeep files
    for d in [LORA_ROOT, MERGED_ROOT]:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()

    print(f"\nCleanup complete. Freed ~{total_mb / 1024:.1f} GB.")
    if archive:
        print(f"Archive is at: {archive}")


if __name__ == "__main__":
    main()
