# Market Sentinel - Apache Deployment Complete! 🎉

## Deployment Summary

Market Sentinel has been successfully deployed to your local Apache server and is fully operational!

## Access Information

### Main Application
**URL:** http://localhost/market_sentinel/

### API Endpoints (Direct Access)
- **Quote:** `http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=quote&symbol=AAPL`
- **Chart:** `http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=chart&symbol=AAPL&interval=1h&range=1d&indicators=ema9,volume`
- **Trades:** `http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=trades&symbol=AAPL&limit=100`

## What Was Deployed

### File Structure
```
/var/www/html/market_sentinel/
├── index.html                          # Frontend web interface
└── cgi-bin/
    ├── api/
    │   └── market_data_api.py         # CGI API endpoint ✅
    ├── molecules/
    │   └── alpaca_molecules/
    │       ├── market_data.py         # Market data module ✅
    │       └── alpaca_config.py       # API credentials ✅
    └── atoms/                         # Supporting modules ✅
        ├── api/
        ├── display/
        └── utils/
```

### Apache Configuration
**File:** `/etc/apache2/conf-available/market_sentinel.conf`

```apache
<Directory "/var/www/html/market_sentinel">
    Options +ExecCGI
    AllowOverride All
    Require all granted
</Directory>

<Directory "/var/www/html/market_sentinel/cgi-bin">
    Options +ExecCGI
    SetHandler cgi-script
    AllowOverride None
    Require all granted
</Directory>

ScriptAlias /market_sentinel/cgi-bin "/var/www/html/market_sentinel/cgi-bin"
```

### Modules Enabled
- ✅ `cgid` (CGI execution for threaded MPM)
- ✅ `headers` (CORS headers)

## Verified Features

### API Endpoints Tested
```bash
# Quote endpoint - WORKING ✅
$ curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=quote&symbol=AAPL' | jq .
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "bid_price": 270.7,
    "ask_price": 270.74,
    "mid_price": 270.72,
    "timestamp": "2025-10-31T16:17:49..."
  }
}

# Chart endpoint with indicators - WORKING ✅
$ curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=chart&symbol=AAPL&interval=1h&range=1d&indicators=ema9,volume' | jq .
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "interval": "1h",
    "bar_count": 16,
    "indicators": {
      "ema9": [...],
      "volume": [...]
    }
  }
}
```

### Frontend Features Available
When you open http://localhost/market_sentinel/ you will have access to:

- 📊 **Real-time candlestick charts** with market data
- 📈 **Technical indicators** (EMA9, EMA21, EMA50, VWAP, Volume)
- ⏰ **Time & Sales** panel with live trades
- 📑 **Multi-tab interface** for multiple symbols
- ⚙️ **Timeframe controls** (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo)
- 📏 **Range selector** (1D, 5D, 1M, 1Y)
- 🔄 **Auto-refresh** (charts: 30s, trades: 10s)
- 🎨 **Professional dark theme** UI

## Testing Commands

### Quick API Tests
```bash
# Test quote
curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=quote&symbol=AAPL' | jq .

# Test chart (1 hour bars, 1 day range)
curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=chart&symbol=AAPL&interval=1h&range=1d' | jq '.data.bar_count'

# Test chart with indicators
curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=chart&symbol=AAPL&interval=1h&range=1d&indicators=ema9,ema21,vwap,volume' | jq '.data.indicators | keys'

# Test trades
curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=trades&symbol=AAPL&limit=10' | jq '.data.count'

# Test different symbols
curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=quote&symbol=MSFT' | jq .
curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=quote&symbol=GOOGL' | jq .
```

### Browser Testing
1. Open: http://localhost/market_sentinel/
2. Enter a symbol (e.g., AAPL, MSFT, GOOGL, TSLA)
3. Watch the candlestick chart load with real data
4. Try different timeframes and ranges
5. Toggle indicators (EMA9, EMA21, VWAP, Volume)
6. View Time & Sales panel for real trades
7. Open multiple tabs for different symbols

## Troubleshooting

### If Charts Don't Load
1. **Check Apache error log:**
   ```bash
   sudo tail -f /var/log/apache2/error.log
   ```

2. **Test API directly:**
   ```bash
   curl 'http://localhost/market_sentinel/cgi-bin/api/market_data_api.py?action=quote&symbol=AAPL'
   ```

3. **Check file permissions:**
   ```bash
   ls -la /var/www/html/market_sentinel/cgi-bin/api/market_data_api.py
   # Should be: -rwxr-xr-x (executable)
   ```

### If You See "500 Internal Server Error"
1. Check Apache error log for details
2. Verify CGI module is enabled: `apache2ctl -M | grep cgi`
3. Check Python path in script: `head -1 /var/www/html/market_sentinel/cgi-bin/api/market_data_api.py`

### If API Returns Empty Data
- Market may be closed (after 4 PM ET)
- Symbol may be invalid
- Try a longer date range (e.g., `range=5d` instead of `range=1d`)

## Apache Commands

```bash
# Restart Apache
sudo systemctl restart apache2

# Check Apache status
sudo systemctl status apache2

# View error log
sudo tail -f /var/log/apache2/error.log

# View access log
sudo tail -f /var/log/apache2/access.log

# Test Apache configuration
sudo apache2ctl configtest

# List enabled modules
apache2ctl -M | grep -E 'cgi|headers'
```

## Redeployment

If you make changes to the code, redeploy using:

```bash
cd /home/wilsonb/dl/github.com/alpaca
./tmp/deploy_to_apache.sh
```

Or manually copy files:
```bash
# Frontend
sudo cp public_html/index.html /var/www/html/market_sentinel/

# Backend
sudo cp cgi-bin/api/market_data_api.py /var/www/html/market_sentinel/cgi-bin/api/
sudo cp cgi-bin/molecules/alpaca_molecules/market_data.py /var/www/html/market_sentinel/cgi-bin/molecules/alpaca_molecules/

# Restart Apache
sudo systemctl reload apache2
```

## Performance Notes

- **Initial load:** ~1-2 seconds for chart data
- **API response time:** <1 second for most requests
- **Auto-refresh:** Charts update every 30 seconds, trades every 10 seconds
- **Concurrent users:** Apache can handle multiple users viewing different symbols

## Security Notes

- ⚠️ API credentials are in `alpaca_config.py` - for paper trading only
- ⚠️ This is configured for local development (CORS: `*`)
- For production: Restrict CORS, use HTTPS, add authentication

## Next Steps

### Optional Enhancements
1. **Custom domain:** Configure Apache virtual host for custom domain
2. **SSL/HTTPS:** Add SSL certificate for secure connections
3. **Authentication:** Add basic auth or OAuth for user login
4. **Rate limiting:** Protect API from abuse
5. **Caching:** Add Redis/memcached for improved performance

### External Access
To access from other devices on your network:

1. Find your local IP:
   ```bash
   ip addr | grep "inet " | grep -v 127.0.0.1
   ```

2. Access from another device:
   ```
   http://<your-ip>/market_sentinel/
   ```

3. May need to configure firewall:
   ```bash
   sudo ufw allow 80/tcp
   ```

## Files for Reference

- **Deployment script:** `/home/wilsonb/dl/github.com/alpaca/tmp/deploy_to_apache.sh`
- **Integration tests:** `/home/wilsonb/dl/github.com/alpaca/tmp/test_integration.py`
- **Full documentation:** `/home/wilsonb/dl/github.com/alpaca/INTEGRATION_COMPLETE.md`

---

**Status:** ✅ **FULLY OPERATIONAL**

**Deployment Date:** 2025-10-31

**Apache Link:** http://localhost/market_sentinel/

**Branch:** `feature/market_sentinel_chart`
