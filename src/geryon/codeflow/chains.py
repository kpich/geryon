"""Chains: separate lines of investigation.

By default every session extends one uninterrupted chain — the generator sees every
hypothesis ever submitted. A chain partitions that: a session tagged ``medonc-pfs``
is shown only ``medonc-pfs`` hypotheses, so a distinct investigation can build
linearly on its own prior work without being flooded by (or polluting) the main line.

A chain is defined by ``chains/<name>.md``: optional frontmatter naming the data
version the chain is valid on, then free prose that is appended to the generator,
critic and narrator system prompts to steer what counts as a good hypothesis.

    ---
    data_version: medonc-pfs-2026-08
    ---

    Prefer hypotheses that contrast the two progression sources...

The file is optional. A chain with no file is a bare label: no focus, no pinned data
version, today's behavior. That is why ``main`` has no ``chains/main.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

DEFAULT_CHAIN = "main"
CHAINS_DIRNAME = "chains"

_VALID_CHAIN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Only keys we act on are recognized; anything else in the frontmatter is ignored so
# a chain file can carry notes for humans without breaking the parser.
_DATA_VERSION_KEY = "data_version"


@dataclass(frozen=True)
class ChainDef:
    """A line of investigation: its label, its steering prose, its data version."""

    name: str
    focus: str | None = None
    data_version: str | None = None


def default_chains_dir() -> Path:
    """``chains/`` under the working directory (matches how geryon_data/ resolves)."""
    return Path.cwd() / CHAINS_DIRNAME


def validate_chain_name(name: str) -> str:
    """Return ``name`` if it is safe to use as a file name, else raise."""
    if not name or not _VALID_CHAIN_NAME.match(name) or name in {".", ".."}:
        raise ValueError(
            f"Invalid chain name {name!r}: use only letters, digits, '.', '_' and "
            f"'-' (no path separators)."
        )
    return name


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split leading ``---`` frontmatter from the body.

    Deliberately not YAML: the only supported shape is ``key: value`` lines, which
    keeps this dependency-free and is all a chain definition needs.
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text  # unterminated fence: treat the whole file as prose

    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip()] = value.strip()

    return meta, "\n".join(lines[end + 1 :]).lstrip("\n")


def chain_path(name: str, chains_dir: Path | str | None = None) -> Path:
    """Path to a chain's definition file (which need not exist)."""
    validate_chain_name(name)
    base = Path(chains_dir) if chains_dir is not None else default_chains_dir()
    return base / f"{name}.md"


def load_chain(name: str, chains_dir: Path | str | None = None) -> ChainDef:
    """Load a chain definition; a missing file yields a bare label."""
    validate_chain_name(name)
    path = chain_path(name, chains_dir)
    if not path.exists():
        return ChainDef(name=name)

    meta, body = _split_frontmatter(path.read_text())
    focus = body.strip() or None
    return ChainDef(
        name=name,
        focus=focus,
        data_version=meta.get(_DATA_VERSION_KEY) or None,
    )
