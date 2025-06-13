#!/bin/bash
# Run fraud prevention pipeline with proper path setup

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the scripts directory
cd "$SCRIPT_DIR/scripts" || exit 1

# Run the pipeline with all arguments passed to this script
python run_pipeline.py "$@"