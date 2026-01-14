"""Workflow runner module.

Can be run as: python -m msk_cycl.workflow.runner
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Literal

from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.workflow import LinearWorkflow, SessionConfig


def get_latest_etl_output(base_dir: Path) -> Path:
    """Find latest ETL output directory by name.

    Parameters
    ----------
    base_dir : Path
        Base directory containing timestamped subdirectories

    Returns
    -------
    Path
        Latest subdirectory (by name sort)

    Raises
    ------
    FileNotFoundError
        If no subdirectories found or base_dir doesn't exist
    """
    base_path = base_dir.expanduser()

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    # Get all subdirectories
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    # Return last one (YYYY-MM-DD format sorts chronologically)
    return subdirs[-1]


def run_workflow(
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    data_base: Path | None = None,
    provider: Literal["ollama", "openai", "anthropic"] = "ollama",
    model: str = "mixtral:8x7b",
    num_proposals: int = 5,
    max_iterations: int = 10,
    enable_llm_logging: bool = True,
) -> list[LabeledHypothesis]:
    """Run hypothesis generation workflow.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory for hypothesis JSONL files
        (default: ~/msk_cycle_hyps/YYYY-MM-DD)
    data_dir : Path, optional
        ETL output directory (default: auto-detect latest from data_base)
    data_base : Path, optional
        Base directory for ETL outputs (default: ~/data/msk_cycle_data)
    provider : Literal["ollama", "openai", "anthropic"]
        LLM provider (default: ollama)
    model : str
        Model name (default: mixtral:8x7b)
    num_proposals : int
        Number of proposals per iteration (default: 5)
    max_iterations : int
        Maximum iterations (default: 10)
    enable_llm_logging : bool
        Enable LLM conversation logging (default: True)

    Returns
    -------
    list[LabeledHypothesis]
        All generated hypotheses
    """
    # Set default output_dir if not provided
    if output_dir is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path.home() / "msk_cycle_hyps" / today

    # Set default data_base if not provided
    if data_base is None:
        data_base = Path.home() / "data" / "msk_cycle_data"

    # Determine data directory
    if data_dir is None:
        parquet_dir = get_latest_etl_output(data_base)
    else:
        parquet_dir = data_dir

    print(f"Using ETL data from: {parquet_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Create config
    config = SessionConfig(
        parquet_dir=parquet_dir,
        storage_dir=output_dir,
        provider_type=provider,
        model=model,
        num_proposals_per_iteration=num_proposals,
        max_iterations=max_iterations,
        enable_llm_logging=enable_llm_logging,
    )

    # Run workflow
    print("Initializing workflow...")
    print()
    workflow = LinearWorkflow(config)
    print()

    print(f"Running full session (up to {max_iterations} iterations)...")
    print()

    hypotheses = workflow.run_full_session()

    print()
    print("✓ Session complete!")
    print(f"  Generated {len(hypotheses)} hypotheses")
    print(f"  Stored in: {output_dir}")

    return hypotheses


def setup_logging(verbose: bool = False) -> None:
    """Set up Python logging.

    Parameters
    ----------
    verbose : bool
        If True, set level to DEBUG; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run hypothesis generation workflow")

    # Output directory (optional, defaults to ~/msk_cycle_hyps/YYYY-MM-DD)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for hypothesis JSONL files "
        "(default: ~/msk_cycle_hyps/YYYY-MM-DD)",
    )

    # Optional args
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        help="ETL output directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--data-base",
        type=Path,
        default=Path.home() / "data" / "msk_cycle_data",
        help="Base directory for ETL outputs (default: ~/data/msk_cycle_data)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default="mixtral:8x7b",
        help="Model name (default: mixtral:8x7b)",
    )
    parser.add_argument(
        "--num-proposals",
        type=int,
        default=5,
        help="Number of proposals per iteration (default: 5)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations (default: 10)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable LLM conversation logging (default: logging enabled)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Set up Python logging
    setup_logging(verbose=args.verbose)

    # Call run_workflow with parsed args
    run_workflow(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        data_base=args.data_base,
        provider=args.provider,
        model=args.model,
        num_proposals=args.num_proposals,
        max_iterations=args.max_iterations,
        enable_llm_logging=not args.no_log,
    )


if __name__ == "__main__":
    main()
