"""Workflow runner module.

Can be run as: python -m msk_cycl.workflow.runner
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Literal

from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.workflow import SessionConfig


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

    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    # Return last one (YYYY-MM-DD format sorts chronologically)
    return subdirs[-1]


def run_workflow(
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    data_base: Path | None = None,
    provider: Literal["openai", "anthropic", "aws_bedrock"] = "aws_bedrock",
    model: str = "us.anthropic.claude-opus-4-5-20251101-v1:0",
    base_url: str | None = None,
    api_key: str | None = None,
    aws_region: str | None = None,
    aws_profile: str | None = None,
    num_proposals: int = 5,
    max_iterations: int = 10,
    enable_llm_logging: bool = True,
    labeled_dir: Path = Path("labeled_hypotheses"),
) -> list[LabeledHypothesis]:
    """Run hypothesis generation workflow.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory for hypothesis JSONL files
        (default: cycl_run_outputs/YYYY-MM-DD)
    data_dir : Path, optional
        ETL output directory (default: auto-detect latest from data_base)
    data_base : Path, optional
        Base directory for ETL outputs (default: ~/data/msk_cycle_data)
    provider : Literal["openai", "anthropic", "aws_bedrock"]
        LLM provider (default: aws_bedrock)
    model : str
        Model name (default: us.anthropic.claude-opus-4-5-20251101-v1:0)
    base_url : str, optional
        Custom API endpoint (e.g., AWS-hosted inference server)
    api_key : str, optional
        API key for authentication
    aws_region : str, optional
        AWS region for Bedrock (default: us-east-1)
    aws_profile : str, optional
        AWS credentials profile name
    num_proposals : int
        Number of proposals per iteration (default: 5)
    max_iterations : int
        Maximum iterations (default: 10)
    enable_llm_logging : bool
        Enable LLM conversation logging (default: True)
    labeled_dir : Path
        Directory with labeled hypothesis JSON files
        (default: labeled_hypotheses/)

    Returns
    -------
    list[LabeledHypothesis]
        All generated hypotheses
    """
    if output_dir is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path.cwd() / "cycl_run_outputs" / today

    output_dir.mkdir(parents=True, exist_ok=True)

    if data_base is None:
        data_base = Path.home() / "data" / "msk_cycle_data"

    if data_dir is None:
        parquet_dir = get_latest_etl_output(data_base)
    else:
        parquet_dir = data_dir

    print(f"Using ETL data from: {parquet_dir}")
    print(f"Output directory: {output_dir}")
    print()

    config = SessionConfig(
        parquet_dir=parquet_dir,
        storage_dir=output_dir,
        provider_type=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        aws_region=aws_region,
        aws_profile=aws_profile,
        num_proposals_per_iteration=num_proposals,
        max_iterations=max_iterations,
        enable_llm_logging=enable_llm_logging,
        labeled_dir=labeled_dir,
    )

    print("Initializing autonomous workflow...")
    print()

    from msk_cycl.workflow.autonomous import AutonomousWorkflow

    workflow = AutonomousWorkflow(config)

    print()

    print(f"Running full session (up to {max_iterations} iterations)...")
    print()

    hypotheses = workflow.run_full_session()

    print()
    print("Session complete!")
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
        "(default: cycl_run_outputs/YYYY-MM-DD)",
    )

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
        choices=["openai", "anthropic", "aws_bedrock"],
        default="aws_bedrock",
        help="LLM provider (default: aws_bedrock)",
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-opus-4-5-20251101-v1:0",
        help="Model name (default: us.anthropic.claude-opus-4-5-20251101-v1:0)",
    )
    parser.add_argument(
        "--base-url",
        help="Custom API endpoint (e.g., https://your-aws-endpoint.com/v1)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (or use env var)",
    )
    parser.add_argument(
        "--aws-region",
        help="AWS region for Bedrock (default: us-east-1)",
    )
    parser.add_argument(
        "--aws-profile",
        help="AWS credentials profile name",
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
        "--labeled-dir",
        type=Path,
        default=Path("labeled_hypotheses"),
        help="Directory with labeled hypothesis JSON files "
        "(default: labeled_hypotheses/)",
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

    setup_logging(verbose=args.verbose)

    run_workflow(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        data_base=args.data_base,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        aws_region=args.aws_region,
        aws_profile=args.aws_profile,
        num_proposals=args.num_proposals,
        max_iterations=args.max_iterations,
        enable_llm_logging=not args.no_log,
        labeled_dir=args.labeled_dir,
    )


if __name__ == "__main__":
    main()
