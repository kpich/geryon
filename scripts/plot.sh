#!/usr/bin/env bash
#
# Plot Pipeline Execution Wrapper
#
# Usage:
#   ./scripts/plot.sh [nextflow options]
#
# Examples:
#   ./scripts/plot.sh                                    # Run with defaults
#   ./scripts/plot.sh --data_dir /path/to/other/data    # Override data location

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NEXTFLOW_DIR="${PROJECT_ROOT}/nextflow"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if nextflow is installed
if ! command -v nextflow &> /dev/null; then
    log_error "Nextflow is not installed or not in PATH"
    log_info "Install from: https://www.nextflow.io/docs/latest/getstarted.html"
    exit 1
fi

# Check if Python package and matplotlib are installed
if ! uv run python -c "import geryon.plot; import matplotlib" 2>/dev/null; then
    log_warn "geryon.plot or matplotlib not found in Python environment"
    log_info "Run: make install (or make dev for development mode)"
    exit 1
fi

# Mirrors params.output_dir in plot.nf; override it the same way.
OUTPUT_DIR="${PROJECT_ROOT}/plots"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --output_dir) OUTPUT_DIR="${args[i + 1]}" ;;
        --output_dir=*) OUTPUT_DIR="${args[i]#--output_dir=}" ;;
    esac
done
# Nextflow resolves a relative path against its launch dir, which is NEXTFLOW_DIR.
[[ "${OUTPUT_DIR}" = /* ]] || OUTPUT_DIR="${NEXTFLOW_DIR}/${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
START_TIME="$(date +%s)"

log_info "Starting plot pipeline..."
log_info "Project root: ${PROJECT_ROOT}"
log_info "Nextflow dir: ${NEXTFLOW_DIR}"

# set -e would exit before the failure message below; capture the code instead.
EXIT_CODE=0
cd "${NEXTFLOW_DIR}"
nextflow run plot.nf \
    -ansi-log true \
    "$@" || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_info "Pipeline completed successfully!"
    INDEX="$(cd "${PROJECT_ROOT}" && uv run python -m geryon.plot.index \
        --output-dir "${OUTPUT_DIR}" --since "${START_TIME}")"
    log_info "Index: ${INDEX}"
    if command -v open &> /dev/null; then
        open "${INDEX}"
    fi
else
    log_error "Pipeline failed with exit code: ${EXIT_CODE}"
fi

exit $EXIT_CODE
