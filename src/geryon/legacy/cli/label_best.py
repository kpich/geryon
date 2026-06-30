"""Rank all hypotheses and write the top N to geryon_data/best.json.

Sends all hypotheses to an LLM ranker and asks it to select the most
scientifically promising ones for follow-up.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from geryon.legacy.cli.annotate import _load_all
from geryon.legacy.labeling.labeled_store import LabeledStore
from geryon.llm.provider import DEFAULT_BEDROCK_MODEL, create_provider
from geryon.llm.providers.base import ChatMessage

RANKER_SYSTEM_PROMPT = """\
You are an expert oncologist reviewing a list of cancer genomics hypotheses.
Each hypothesis compares two cohorts on a survival or other outcome.

Your task: select the TOP {n} hypotheses most worth pursuing further, based on
scientific promise. "Promising" means: statistically credible, clinically
meaningful (not just a trivially known association), and actionable — i.e., the
finding suggests a real biological difference or treatment-relevant subgroup.

Return ONLY valid JSON with no prose before or after it. Use the full ID from
the "ID:" field, not the short 8-character prefix from the heading:
{{
  "top_ids": ["<full hypothesis_id>", ...],
  "rationale": "1-3 sentence explanation of what made these stand out"
}}"""


def _build_user_prompt(hyps: list[dict]) -> str:
    sections: list[str] = []
    for h in hyps:
        hid = h["hypothesis_id"]

        hr = h.get("hazard_ratio")
        ci_lo = h.get("confidence_interval_lower")
        ci_hi = h.get("confidence_interval_upper")
        pv = h.get("p_value")
        na = h.get("cohort_a_size")
        nb = h.get("cohort_b_size")

        stats = f"HR={hr:.3f}" if hr is not None else "HR=N/A"
        if ci_lo is not None:
            stats += f", 95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]"
        if pv is not None:
            stats += f", p={pv:.4f}"
        stats += f", N_A={na}, N_B={nb}"

        limitations = h.get("limitations") or []
        if isinstance(limitations, list):
            lim_str = "; ".join(limitations)
        else:
            lim_str = str(limitations)

        section = (
            f"## Hypothesis {hid[:8]}\n"
            f"ID: {hid}\n"
            f"Cohort A: {h.get('cohort_a_description')}\n"
            f"Cohort B: {h.get('cohort_b_description')}\n"
            f"Rationale: {h.get('rationale')}\n"
            f"Stats: {stats}\n"
            f"Summary: {h.get('summary')}\n"
            f"Findings: {h.get('findings')}\n"
            f"Limitations: {lim_str}"
        )
        sections.append(section)

    return "Select the top hypotheses below. Return JSON.\n\n" + "\n\n".join(sections)


def _parse_response(content: str, hyps: list[dict], n: int) -> tuple[list[str], str]:
    """Return (top_ids, rationale), falling back to first N on parse failure."""
    # Build prefix→full-id lookup so short IDs from the LLM still resolve
    prefix_to_id = {h["hypothesis_id"][:8]: h["hypothesis_id"] for h in hyps}

    try:
        text = content.strip()
        # Find the JSON block even when preceded by prose
        fence_start = text.find("```")
        if fence_start != -1:
            lines = text[fence_start:].split("\n")
            inner_lines = []
            for line in lines[1:]:  # skip opening fence
                if line.strip().startswith("```"):
                    break
                inner_lines.append(line)
            text = "\n".join(inner_lines)

        data = json.loads(text)
        raw_ids = data["top_ids"]
        top_ids = [prefix_to_id.get(rid, rid) for rid in raw_ids]
        rationale = data.get("rationale", "")
        return top_ids, rationale
    except Exception as e:
        print(f"[label_best] parse error: {e}")
        print(f"[label_best] raw response:\n{content[:500]}")
        fallback_ids = [h["hypothesis_id"] for h in hyps[:n]]
        return fallback_ids, "parse error — first N selected"


def main():
    parser = argparse.ArgumentParser(
        description="Rank hypotheses by scientific promise and write best.json"
    )
    parser.add_argument(
        "--output-dir",
        default="geryon_data/sessions",
        help="Directory containing session JSONL files "
        "(default: geryon_data/sessions/)",
    )
    parser.add_argument(
        "--labeled-dir",
        default="geryon_data/labeled",
        help="Directory for labeled hypothesis JSON files "
        "(default: geryon_data/labeled/)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of top hypotheses to select (default: 5)",
    )
    parser.add_argument(
        "--out",
        default="geryon_data/best.json",
        help="Output path for best.json (default: geryon_data/best.json)",
    )
    parser.add_argument(
        "--provider",
        default="aws_bedrock",
        help="LLM provider (default: aws_bedrock)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BEDROCK_MODEL,
        help="Model ID",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
        help="AWS profile (optional)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labeled_store = LabeledStore(Path(args.labeled_dir))

    hyps = _load_all(output_dir, labeled_store)
    if not hyps:
        print("No hypotheses found.")
        return

    system_prompt = RANKER_SYSTEM_PROMPT.format(n=args.n)
    user_prompt = _build_user_prompt(hyps)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    provider_kwargs = {"model": args.model, "region": args.aws_region}
    if args.aws_profile:
        provider_kwargs["profile"] = args.aws_profile

    provider = create_provider(args.provider, **provider_kwargs)
    response = provider.generate(messages, temperature=0.3, cache_system=True)

    top_ids, rationale = _parse_response(response.content, hyps, args.n)

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n": args.n,
        "hypothesis_ids": top_ids,
        "rationale": rationale,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(out_path)
    print("Top IDs:")
    for hid in top_ids:
        print(f"  {hid}")


if __name__ == "__main__":
    main()
