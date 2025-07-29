# EMA Divergence Quality Prediction PRD

## Overview
Real-time prediction system to identify high-quality superduper alerts with higher EMA divergences without requiring full trading day data for median calculations.

## Problem Statement
Current system uses intraday mean EMA divergence to filter alerts, but this requires waiting for sufficient alerts to calculate an accurate mean. We need a predictive approach that can identify high-quality alerts (those likely to have above-mean EMA divergence) in real-time.

## Success Criteria
- Identify 80%+ of alerts that would have been classified as "above mean EMA divergence" 
- Work generically across different stocks without stock-specific tuning
- Operate in real-time without requiring full day's alert data

## Data Analysis Requirements

### Historical Data Collection
**IMPORTANT**: Only use data from 2025-07-28 onwards for analysis. Earlier data may not reflect current market conditions or alert generation algorithms.

Collect and analyze the following metrics from historical superduper alerts:
- EMA divergence values (primary target)
- Volume ratios 
- Confidence scores
- Stock symbols
- Alert timestamps
- Market volatility indicators (if available)

### Analysis Goals
1. **Establish Universal Threshold**: Determine fixed EMA divergence threshold that works across stocks
2. **Identify Leading Indicators**: Find volume ratio and confidence score thresholds that predict high EMA divergence
3. **Volatility Adjustment**: Determine if stock volatility affects optimal thresholds

## Proposed Solution

### Multi-Factor Quality Score
Implement composite scoring system:

```
Quality_Score = (EMA_divergence × 0.5) + (volume_ratio × 0.2) + (confidence × 0.3)
```

### Primary Classification Logic
```
High_Quality_Alert = (
    EMA_divergence >= FIXED_THRESHOLD OR
    (volume_ratio >= VOLUME_THRESHOLD AND confidence >= CONFIDENCE_THRESHOLD)
)
```

### Initial Threshold Estimates
Based on 2025-07-28 VWAV data:
- **EMA Divergence Threshold**: 3.0% (baseline)
- **Volume Ratio Threshold**: 2.7x
- **Confidence Threshold**: 0.91

## Implementation Requirements

### Phase 1: Historical Analysis
1. **Data Collection**: Aggregate all superduper alert data from 2025-07-28 onwards
2. **Statistical Analysis**: 
   - Calculate percentile distributions for EMA divergence across all stocks
   - Analyze correlation between volume ratio, confidence, and EMA divergence
   - Identify optimal thresholds using ROC analysis
3. **Validation**: Test thresholds on held-out data to measure accuracy

### Phase 2: Production Integration
1. **Real-time Scoring**: Implement quality score calculation in alert pipeline
2. **Threshold Configuration**: Make thresholds configurable for easy adjustment
3. **Monitoring**: Track prediction accuracy vs actual results
4. **Feedback Loop**: Weekly threshold adjustment based on performance metrics

## Technical Specifications

### Configuration Parameters
```python
EMA_DIVERGENCE_THRESHOLD = 0.030  # 3.0%
VOLUME_RATIO_THRESHOLD = 2.7
CONFIDENCE_THRESHOLD = 0.91
QUALITY_SCORE_THRESHOLD = 2.5  # Alternative composite approach
```

### Alert Enhancement
Add quality prediction fields to alert structure:
```json
{
    "predicted_quality": "HIGH" | "MEDIUM" | "LOW",
    "quality_score": 2.85,
    "quality_factors": {
        "ema_divergence_score": 1.8,
        "volume_score": 0.54,
        "confidence_score": 0.51
    }
}
```

## Success Metrics
- **Precision**: % of predicted high-quality alerts that are actually above mean
- **Recall**: % of actual above-mean alerts that are predicted as high-quality
- **F1 Score**: Balanced measure of precision and recall
- **False Positive Rate**: % of below-mean alerts incorrectly classified as high-quality

## Target Performance
- Precision: ≥80%
- Recall: ≥75%
- F1 Score: ≥77%
- False Positive Rate: ≤20%

## Risk Mitigation
1. **Market Regime Changes**: Monitor threshold performance and adjust quarterly
2. **Stock-Specific Variations**: Consider symbol-based threshold adjustments if needed
3. **Volume Anomalies**: Implement volume ratio caps to prevent extreme outliers
4. **Confidence Score Reliability**: Validate confidence score calculation consistency

## Timeline
- **Week 1-2**: Historical data collection and analysis
- **Week 3**: Threshold optimization and validation
- **Week 4**: Production implementation and testing
- **Ongoing**: Performance monitoring and adjustments

## Dependencies
- Access to historical superduper alert data from 2025-07-28 onwards
- Statistical analysis tools (pandas, numpy, scikit-learn)
- Alert pipeline modification capabilities
- Performance monitoring infrastructure