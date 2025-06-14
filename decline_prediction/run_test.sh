#!/bin/bash
# Run comprehensive test of decline prediction system

echo "====================================="
echo "Decline Prediction System Test"
echo "====================================="

# Change to decline_prediction directory
cd "$(dirname "$0")"

# Create necessary directories
mkdir -p logs results models data/processed data/graph scripts api cache utils config

# Run the test
python test_decline_system.py

# Check if report was generated
if [ -f "results/management_report.md" ]; then
    echo ""
    echo "Test completed! Management report available at:"
    echo "  results/management_report.md"
    echo ""
    echo "To view the report:"
    echo "  cat results/management_report.md"
    echo ""
else
    echo "Test may have failed - no report generated"
fi