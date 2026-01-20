"""List available hypothesis sessions for review."""

import json
from pathlib import Path


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="List hypothesis sessions")
    parser.add_argument("--storage-dir", required=True, help="Storage directory")
    args = parser.parse_args()

    storage_path = Path(args.storage_dir)

    if not storage_path.exists():
        print(f"Storage directory does not exist: {storage_path}")
        return

    # Find all session JSONL files
    session_files = list(storage_path.rglob("*.jsonl"))

    if not session_files:
        print(f"No session files found in {storage_path}")
        return

    print("Available sessions:")
    print("-" * 80)

    for file in sorted(session_files):
        try:
            # Load session metadata
            with open(file) as f:
                first_line = f.readline()
                metadata = json.loads(first_line)

                if metadata.get("record_type") == "metadata":
                    session_id = metadata.get("session_id", "unknown")
                    created_at = metadata.get("created_at", "unknown")

                    # Count hypotheses (subtract 1 for metadata line)
                    num_hyps = sum(1 for _ in f)

                    print(f"Session: {session_id}")
                    print(f"  Created: {created_at}")
                    print(f"  Hypotheses: {num_hyps}")
                    print(f"  File: {file}")
                    print()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue


if __name__ == "__main__":
    main()
