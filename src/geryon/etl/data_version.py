"""Human-readable identity for an ETL output directory.

An ETL run publishes to ``<output_base>/<version>/``. The version name is chosen by
a human (``medonc-pfs-2026-08``), not derived from the run date, because it is what
gets recorded on every hypothesis and named in a chain definition — remembering which
anodyne date held which cohort is exactly the failure mode this avoids. The date
remains the default so an unnamed run behaves as it always has.

``VERSION.json`` sits at the version-dir root and is copied into each split subdir by
``split_by_patient``, so an ``explore/`` directory is self-describing on its own.
Legacy dated dirs have no marker; ``read_data_version`` returns None for them and
callers fall back to the directory name.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re

VERSION_MARKER_FILENAME = "VERSION.json"

_VALID_VERSION_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Named locally rather than imported from split_by_patient, which imports this module.
_SPLIT_DIR_NAMES = frozenset({"explore", "validation"})


def validate_version_name(name: str) -> str:
    """Return ``name`` if it is safe to use as a directory name, else raise."""
    if not name or not _VALID_VERSION_NAME.match(name) or name in {".", ".."}:
        raise ValueError(
            f"Invalid data version name {name!r}: use only letters, digits, "
            f"'.', '_' and '-' (no path separators)."
        )
    return name


def write_version_marker(
    version_dir: Path | str,
    *,
    name: str,
    data_root: str,
    holdout_seed: int,
) -> Path:
    """Write ``VERSION.json`` identifying an ETL output directory."""
    validate_version_name(name)
    marker = Path(version_dir) / VERSION_MARKER_FILENAME
    marker.write_text(
        json.dumps(
            {
                "name": name,
                "created_at": datetime.now(UTC).isoformat(),
                "data_root": data_root,
                "holdout_seed": holdout_seed,
            },
            indent=2,
        )
        + "\n"
    )
    return marker


def read_data_version(version_dir: Path | str) -> str | None:
    """Return the version name a directory was published under, or None if unmarked."""
    marker = Path(version_dir) / VERSION_MARKER_FILENAME
    if not marker.exists():
        return None
    try:
        name = json.loads(marker.read_text()).get("name")
    except (OSError, json.JSONDecodeError):
        return None
    return str(name) if name else None


def resolve_data_version(parquet_dir: Path | str) -> str:
    """Best-effort version name for a parquet dir, for stamping onto hypotheses.

    Falls back to the enclosing directory name so legacy dated outputs (which have no
    marker) still record which cohort a hypothesis ran against.
    """
    parquet_dir = Path(parquet_dir)
    name = read_data_version(parquet_dir)
    if name:
        return name
    # A split subdir (explore/) inherits its identity from the version dir above it.
    name = read_data_version(parquet_dir.parent)
    if name:
        return name
    if parquet_dir.name in _SPLIT_DIR_NAMES:
        return parquet_dir.parent.name
    return parquet_dir.name


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Write a VERSION.json marker into an ETL output directory."
    )
    parser.add_argument("--dir", default=".", help="Directory to mark (default: cwd)")
    parser.add_argument("--name", required=True, help="Human-readable version name")
    parser.add_argument("--data-root", required=True, help="Source TSV directory")
    parser.add_argument("--holdout-seed", type=int, required=True)
    args = parser.parse_args()

    write_version_marker(
        args.dir,
        name=args.name,
        data_root=args.data_root,
        holdout_seed=args.holdout_seed,
    )


if __name__ == "__main__":
    main()
