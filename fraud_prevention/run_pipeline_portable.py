#!/usr/bin/env python3
"""
Portable pipeline runner that handles different execution contexts
"""

import os
import sys
import subprocess

# Determine the fraud_prevention directory
if os.path.exists('fraud_prevention'):
    # Running from parent directory
    fraud_dir = 'fraud_prevention'
elif os.path.exists('scripts') and os.path.exists('config'):
    # Running from fraud_prevention directory
    fraud_dir = '.'
elif os.path.exists('../scripts') and os.path.exists('../config'):
    # Running from a subdirectory
    fraud_dir = '..'
else:
    # Try to find it relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'fraud_prevention' in script_dir:
        fraud_dir = script_dir[:script_dir.index('fraud_prevention') + len('fraud_prevention')]
    else:
        print("Error: Cannot find fraud_prevention directory")
        sys.exit(1)

# Change to fraud_prevention directory
os.chdir(fraud_dir)
print(f"Working directory: {os.getcwd()}")

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))

# Import and run the main pipeline
try:
    os.chdir('scripts')
    import run_pipeline
    
    # Pass command line arguments
    original_argv = sys.argv
    sys.argv = ['run_pipeline.py'] + sys.argv[1:]
    
    # Run the pipeline
    exit_code = run_pipeline.main()
    sys.exit(exit_code)
    
except Exception as e:
    print(f"Error running pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)