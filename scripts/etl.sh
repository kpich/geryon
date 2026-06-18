#!/usr/bin/env bash
#
# ETL Pipeline Execution Wrapper
#
# Usage:
#   ./scripts/etl.sh [nextflow options]
#
# Examples:
#   ./scripts/etl.sh                                    # Run with defaults
#   ./scripts/etl.sh --data_root /path/to/other/data   # Override data location

set -euo pipefail

# STOPGAP: force Nextflow's legacy (v1) language parser. etl.nf still uses old
# syntax (e.g. top-level `workflow.onComplete { ... }`) that the strict parser
# in Nextflow 26+ rejects. Remove once etl.nf is migrated to the new syntax.
export NXF_SYNTAX_PARSER=v1

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

# Check if Python package is installed
if ! python -c "import geryon.etl" 2>/dev/null; then
    log_warn "geryon package not found in Python environment"
    log_info "Run: make install (or make dev for development mode)"
    exit 1
fi

# Navigate to Nextflow directory
cd "${NEXTFLOW_DIR}"

# Run pipeline
log_info "Starting ETL pipeline..."
log_info "Project root: ${PROJECT_ROOT}"
log_info "Nextflow dir: ${NEXTFLOW_DIR}"

nextflow run etl.nf \
    -ansi-log true \
    "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_info "Pipeline completed successfully!"
else
    log_error "Pipeline failed with exit code: ${EXIT_CODE}"
fi

exit $EXIT_CODE
