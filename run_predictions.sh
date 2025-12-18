#!/bin/bash

# Run predictions for all percent thresholds
# Usage:
#   ./run_predictions.sh                              # Uses 2025-12-17 as default
#   ./run_predictions.sh 2025-12-17                   # Predict on single date
#   ./run_predictions.sh 2025-12-17 2025-12-18        # Predict on date range

# Get start and end dates from arguments
START_DATE="${1:-2025-12-17}"  # Default to 2025-12-17 if not provided
END_DATE="${2:-$START_DATE}"   # Default to start_date if not provided

echo "Running predictions for all models..."
echo "Date Range: $START_DATE to $END_DATE"
echo

for percent in 1.5 2 2.5 3 4 5 6 7; do
    echo "================================================"
    echo "Running predictions for ${percent}% model"
    echo "================================================"
    python analysis/predict_squeeze_outcomes.py predict \
        --model analysis/xgboost_model_${percent}pct.json \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"
    echo
done

echo "All predictions complete!"
