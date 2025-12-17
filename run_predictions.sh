#!/bin/bash

# Run predictions for all percent thresholds
# Usage: ./run_predictions.sh

echo "Running predictions for all models..."
echo

for percent in 1.5 2 2.5 3 4 5 6 7; do
    echo "================================================"
    echo "Running predictions for ${percent}% model"
    echo "================================================"
    python analysis/predict_squeeze_outcomes.py predict \
        --model analysis/xgboost_model_${percent}pct.json \
        --test-dir historical_data/2025-12-16
    echo
done

echo "All predictions complete!"
