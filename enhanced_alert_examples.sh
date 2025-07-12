#!/bin/bash
# Enhanced ORB Alert Test Runner - Usage Examples

echo "🧪 Enhanced ORB Alert Test Runner - Usage Examples"
echo "=================================================="
echo ""

echo "📊 Basic Usage:"
echo "python run_enhanced_alert_test.py DATE [OPTIONS]"
echo ""

echo "📅 Examples:"
echo ""

echo "1. Test all symbols for a specific date:"
echo "   python run_enhanced_alert_test.py 2025-07-11"
echo "   python run_enhanced_alert_test.py 2025-07-11 --all"
echo ""

echo "2. Test a specific symbol:"
echo "   python run_enhanced_alert_test.py 2025-07-11 --symbol FTFT"
echo "   python run_enhanced_alert_test.py 2025-07-10 --symbol PROK"
echo ""

echo "3. Test multiple specific symbols:"
echo "   python run_enhanced_alert_test.py 2025-07-11 --symbols FTFT,PROK,AAPL"
echo "   python run_enhanced_alert_test.py 2025-07-10 --symbols SAFX,MLGO,KLTO"
echo ""

echo "4. Skip chart generation for faster processing:"
echo "   python run_enhanced_alert_test.py 2025-07-11 --all --no-charts"
echo ""

echo "🔍 What the script does:"
echo "- Loads historical market data for the specified date"
echo "- Calculates ORB (Opening Range Breakout) features"
echo "- Applies PCA-derived filters for high-quality setups"
echo "- Detects bullish breakouts and bearish breakdowns"
echo "- Generates confidence scores based on volume, momentum, and range"
echo "- Creates candlestick charts with alert timing visualization"
echo "- Saves results as JSON files for further analysis"
echo ""

echo "📊 PCA Filters Applied:"
echo "- Volume Ratio: Must be > 2.5x average"
echo "- Duration: Must be > 10 minutes"
echo "- Momentum: Must be > -0.01"
echo "- Range: Must be between 5.0% and 35.0%"
echo ""

echo "📁 Output Files:"
echo "- test_results/enhanced_alerts_YYYY-MM-DD/"
echo "  ├── SYMBOL_enhanced_realtime_alert_test.png  (Chart visualization)"
echo "  ├── SYMBOL_enhanced_realtime_alert_test.pdf  (PDF version)"
echo "  ├── SYMBOL_test_results.json                 (Individual results)"
echo "  └── test_summary.json                        (Overall summary)"
echo ""

echo "✅ Recent successful tests:"
echo "- FTFT on 2025-07-11: Enhanced Bullish Breakout at 09:46:00 ET (100% confidence)"
echo "- PROK on 2025-07-10: Enhanced Bullish Breakout at 13:45:00 ET (80% confidence)"
echo "- 2025-07-11 All Symbols: 10 alerts generated across 7 symbols"