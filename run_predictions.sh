#!/bin/bash

# Run predictions for all percent thresholds
# Usage:
#   ./run_predictions.sh                              # Uses 2025-12-17 as default, 09:30-12:00 time range
#   ./run_predictions.sh 2025-12-17                   # Predict on single date, 09:30-12:00 time range
#   ./run_predictions.sh 2025-12-17 2025-12-18        # Predict on date range, 09:30-12:00 time range
#   ./run_predictions.sh 2025-12-17 2025-12-17 10:00 15:00  # Custom time range

# Get start and end dates from arguments
START_DATE="${1:-2025-12-17}"  # Default to 2025-12-17 if not provided
END_DATE="${2:-$START_DATE}"   # Default to start_date if not provided
START_TIME="${3:-09:30}"       # Default to 09:30 if not provided
END_TIME="${4:-12:00}"         # Default to 12:00 if not provided

echo "Running predictions for all models..."
echo "Date Range: $START_DATE to $END_DATE"
echo "Time Range: $START_TIME to $END_TIME"
echo

for percent in 1.5 2 2.5 3 4 5 6 7; do
    echo "================================================"
    echo "Running predictions for ${percent}% model"
    echo "================================================"
    python analysis/predict_squeeze_outcomes.py predict \
        --model analysis/xgboost_model_${percent}pct.json \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --start-time "$START_TIME" \
        --end-time "$END_TIME" \
        --prediction-threshold 0.5
    echo
done

echo "All predictions complete!"
