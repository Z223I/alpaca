---
description: "Analyze recent ORB alerts and trading data performance"
allowed-tools: ["bash", "read", "ls"]
---

# Analyze ORB Alerts Performance

Analyze recent alerts and trading data to understand system performance.

$ARGUMENTS

## Alert Summary for Today
```bash
!echo "=== Alert Summary $(date +%Y-%m-%d) ==="
!echo "Bullish alerts: $(ls alerts/bullish/*$(date +%Y%m%d)* 2>/dev/null | wc -l)"
!echo "Bearish alerts: $(ls alerts/bearish/*$(date +%Y%m%d)* 2>/dev/null | wc -l)"
```

## Recent High-Priority Alerts
```bash
!echo "=== Recent High-Priority Alerts ==="
!find alerts/ -name "*.json" -mtime -1 -exec grep -l '"priority": "HIGH"' {} \; | head -5 | while read file; do
    echo "📈 $file"
    grep -E '"symbol"|"current_price"|"breakout_type"' "$file" | head -3
    echo "---"
done 2>/dev/null || echo "No recent high-priority alerts found"
```

## Data Collection Status
```bash
!echo "=== Data Collection Status ==="
!if [ -d "historical_data/$(date +%Y-%m-%d)" ]; then
    echo "✅ Today's data directory exists"
    echo "Market data files: $(ls historical_data/$(date +%Y-%m-%d)/market_data/*.csv 2>/dev/null | wc -l)"
    echo "Latest file: $(ls -t historical_data/$(date +%Y-%m-%d)/market_data/*.csv 2>/dev/null | head -1 | xargs basename)"
else
    echo "❌ No data directory for today"
fi
```

## Symbol Performance Overview
```bash
!echo "=== Most Active Symbols (by alert count) ==="
!find alerts/ -name "*.json" -mtime -7 | xargs grep -h '"symbol"' 2>/dev/null | cut -d'"' -f4 | sort | uniq -c | sort -nr | head -5 || echo "No recent alerts to analyze"
```