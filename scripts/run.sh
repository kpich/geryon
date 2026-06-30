#!/usr/bin/env bash
set -euo pipefail

# Run the workflow module directly with all arguments
exec uv run python -u -m geryon.legacy.workflow.runner "$@"
