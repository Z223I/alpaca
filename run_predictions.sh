#!/bin/bash

# Run predictions for all percent thresholds
# Usage:
#   ./run_predictions.sh                                        # Default: 2025-12-17, 09:30-12:00, no price filter
#   ./run_predictions.sh 2025-12-17                             # Single date, 09:30-12:00 time range
#   ./run_predictions.sh 2025-12-17 2025-12-18                  # Date range, 09:30-12:00 time range
#   ./run_predictions.sh 2025-12-17 2025-12-17 10:00 15:00      # Single date with custom time range
#   ./run_predictions.sh 2025-12-17 2025-12-18 09:30 12:00 2 10 # Date range, time range, price filter ($2-$10)
#   ./run_predictions.sh 2025-12-17 2025-12-17 09:30 12:00 5    # Only minimum price filter ($5+)
#   ./run_predictions.sh 2025-12-17 2025-12-17 09:30 12:00 "" 20 # Only maximum price filter (up to $20)

# Get start and end dates from arguments
START_DATE="${1:-2025-12-17}"  # Default to 2025-12-17 if not provided
END_DATE="${2:-$START_DATE}"   # Default to start_date if not provided
START_TIME="${3:-09:30}"       # Default to 09:30 if not provided
END_TIME="${4:-12:00}"         # Default to 12:00 if not provided
MIN_PRICE="${5:-}"             # No default - optional
MAX_PRICE="${6:-}"             # No default - optional

echo "Running predictions for all models..."
echo "Date Range: $START_DATE to $END_DATE"
echo "Time Range: $START_TIME to $END_TIME"
if [ -n "$MIN_PRICE" ] || [ -n "$MAX_PRICE" ]; then
    echo "Price Range: \$${MIN_PRICE:-any} to \$${MAX_PRICE:-any}"
fi
echo
echo "NOTE: Models 1.5% and 6.0% use REALISTIC labels (hit target before stop)."
echo "      Other models use MAX_GAIN labels (eventually hit target)."
echo

for percent in 1.5 2 2.5 3 4 5 6 7; do
    echo "================================================"
    echo "Running predictions for ${percent}% model"
    echo "================================================"

    # Build command with optional price filters
    CMD="python analysis/predict_squeeze_outcomes.py predict \
        --model analysis/tuned_models/xgboost_tuned_${percent}pct.json \
        --start-date \"$START_DATE\" \
        --end-date \"$END_DATE\" \
        --start-time \"$START_TIME\" \
        --end-time \"$END_TIME\" \
        --prediction-threshold 0.5"

    # Add price filters if provided
    if [ -n "$MIN_PRICE" ]; then
        CMD="$CMD --min-price $MIN_PRICE"
    fi
    if [ -n "$MAX_PRICE" ]; then
        CMD="$CMD --max-price $MAX_PRICE"
    fi

    # Execute command
    eval $CMD
    echo
done

echo "All predictions complete!"
