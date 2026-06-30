"""Code-first workflow runner.

Run as: python -m geryon.codeflow.runner
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import uuid

from geryon.codeflow.agent import CodeWorkflow
from geryon.codeflow.models import CodeHypothesis
from geryon.etl.split_by_patient import EXPLORE_SPLIT, SPLIT_MARKER_FILENAME
from geryon.llm import DEFAULT_BEDROCK_MODEL
from geryon.workflow.session import SessionConfig


def get_latest_etl_output(base_dir: Path) -> Path:
    """Find the latest ETL output dir by name (YYYY-MM-DD sorts chronologically)."""
    base_path = base_dir.expanduser()
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")
    return subdirs[-1]


def resolve_explore_dir(path: Path) -> Path:
    """Resolve to the exploration-split parquet dir, enforcing the holdout.

    Accepts either a dated ETL dir (which must contain an ``explore/`` subdir) or a
    dir that is already the exploration split (identified by its ``SPLIT`` marker).
    Hard-fails on a legacy, un-split dir so the inner loop can never silently read
    the full cohort — that validation leak is exactly the bug this guards against.
    """
    path = Path(path)
    marker = path / SPLIT_MARKER_FILENAME
    if marker.exists():
        got = marker.read_text().strip()
        if got != EXPLORE_SPLIT:
            raise SystemExit(
                f"Refusing to run on the '{got}' split — the inner loop must read "
                f"the '{EXPLORE_SPLIT}' split only: {path}"
            )
        return path

    candidate = path / EXPLORE_SPLIT
    if (candidate / SPLIT_MARKER_FILENAME).exists():
        return candidate

    raise SystemExit(
        f"No '{EXPLORE_SPLIT}/' split subdir found under {path}. This ETL output "
        f"predates the data-layer holdout split. Re-run the ETL (nextflow/etl.nf) "
        f"so the validation cohort is physically absent from the inner loop."
    )


def run_workflow(
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    data_base: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    aws_region: str | None = None,
    aws_profile: str | None = None,
    num_proposals: int | None = None,
    max_iterations: int | None = None,
    critic_cycles: int | None = None,
    sandbox_timeout: int | None = None,
    enable_llm_logging: bool | None = None,
) -> list[CodeHypothesis]:
    """Run the code-first hypothesis generation workflow."""
    session_id = str(uuid.uuid4())
    today = datetime.now().strftime("%Y-%m-%d")

    if output_dir is None:
        output_dir = Path.cwd() / "geryon_data" / "code_sessions"
    run_dir = output_dir / today / session_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if data_base is None:
        data_base = Path.home() / "data" / "geryon_data"
    etl_dir = data_dir if data_dir is not None else get_latest_etl_output(data_base)
    parquet_dir = resolve_explore_dir(etl_dir)

    print(f"Using ETL data from: {parquet_dir}")
    print(f"Output directory: {run_dir}")
    print()

    overrides: dict = {
        "provider_type": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "aws_region": aws_region,
        "aws_profile": aws_profile,
        "num_proposals_per_iteration": num_proposals,
        "max_iterations": max_iterations,
        "critic_cycles": critic_cycles,
        "sandbox_timeout_seconds": sandbox_timeout,
        "enable_llm_logging": enable_llm_logging,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config = SessionConfig(
        session_id=session_id,
        parquet_dir=parquet_dir,
        storage_dir=run_dir,
        output_dir=output_dir,
        **overrides,
    )
    (run_dir / "config.json").write_text(json.dumps(config.to_config_dict(), indent=2))

    print("Initializing code-first workflow...")
    workflow = CodeWorkflow(config)
    print(f"Running full session (up to {config.max_iterations} iterations)...\n")
    hypotheses = workflow.run_full_session()

    print(f"\nSession complete! Generated {len(hypotheses)} hypotheses in {run_dir}")
    return hypotheses


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code-first hypothesis generation")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("-d", "--data-dir", type=Path, default=None)
    parser.add_argument(
        "--data-base", type=Path, default=Path.home() / "data" / "geryon_data"
    )
    parser.add_argument(
        "--provider", choices=["openai", "anthropic", "aws_bedrock"], default=None
    )
    parser.add_argument(
        "--model", default=None, help=f"default: {DEFAULT_BEDROCK_MODEL}"
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--aws-region", default=None)
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--num-proposals", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--critic-cycles",
        type=int,
        default=None,
        help="Run the agentic critic on each hypothesis when > 0 (default 0)",
    )
    parser.add_argument("--sandbox-timeout", type=int, default=None)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
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
        critic_cycles=args.critic_cycles,
        sandbox_timeout=args.sandbox_timeout,
        enable_llm_logging=False if args.no_log else None,
    )


if __name__ == "__main__":
    main()
