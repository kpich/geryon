#!/usr/bin/env bash
set -euo pipefail

# Parse arguments to check if using ollama provider
PROVIDER="ollama"  # default
for arg in "$@"; do
    if [[ "$arg" =~ --provider[=\ ](.+) ]]; then
        PROVIDER="${BASH_REMATCH[1]}"
    fi
done

# Only check/start Ollama if using ollama provider
if [[ "$PROVIDER" == "ollama" ]]; then
    echo "Checking if Ollama is running..."

    # Check if Ollama is responding
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is not running. Starting Ollama..."

        # Check if ollama command exists
        if ! command -v ollama &> /dev/null; then
            echo "ERROR: ollama command not found. Please install Ollama first:"
            echo "  https://ollama.ai"
            exit 1
        fi

        # Start Ollama in background
        ollama serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!

        echo "Waiting for Ollama to start (PID: $OLLAMA_PID)..."

        # Wait for Ollama to be ready (max 30 seconds)
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo "Ollama is ready!"
                break
            fi

            if [ $i -eq 30 ]; then
                echo "ERROR: Ollama failed to start after 30 seconds"
                echo "Check logs at /tmp/ollama.log"
                exit 1
            fi

            sleep 1
        done
    else
        echo "Ollama is already running"
    fi
fi

# Run the workflow module directly with all arguments
exec python -m msk_cycl.workflow.runner "$@"
