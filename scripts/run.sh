#!/usr/bin/env bash
set -euo pipefail

# Run the workflow module directly with all arguments
exec uv run python -m msk_cycl.workflow.runner "$@"
