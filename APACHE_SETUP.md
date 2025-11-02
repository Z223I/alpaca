# Apache Server Setup - Complete

## ✅ Setup Completed

I've successfully configured the Apache server to serve Market Sentinel:

### 1. Created Symlink: `~/public_html` → Repository
```bash
~/public_html -> /home/wilsonb/dl/github.com/alpaca/public_html
```

### 2. Created Symlink: `public_html/cgi-bin` → Repository CGI
```bash
~/public_html/cgi-bin -> /home/wilsonb/dl/github.com/alpaca/cgi-bin
```

### 3. Verified Apache Configuration
- ✅ **UserDir module:** Enabled (`userdir_module`)
- ✅ **CGI module:** Enabled (`cgid_module`)
- ✅ **CGI .htaccess:** Properly configured in `cgi-bin/.htaccess`
- ✅ **File permissions:** Correct (755 for directories, 644 for HTML, 755 for CGI scripts)
- ✅ **ExecCGI:** Enabled via `.htaccess`

## Access URLs

### Main Interface:
```
http://localhost/~wilsonb/index.html
```
or
```
http://localhost/~wilsonb/
```

### CGI API Endpoints:
```
http://localhost/~wilsonb/cgi-bin/api/market_data_api.py?action=quote&symbol=AAPL
http://localhost/~wilsonb/cgi-bin/api/market_data_api.py?action=chart&symbol=AAPL&interval=1m&range=1d
http://localhost/~wilsonb/cgi-bin/api/market_data_api.py?action=trades&symbol=AAPL&limit=100
```

## Directory Structure

```
~/public_html/                           (symlink to repo)
├── index.html                           Market Sentinel web interface
└── cgi-bin/                             (symlink to repo)
    ├── .htaccess                        CGI configuration
    ├── api/
    │   ├── market_data_api.py           Main API endpoint
    │   └── market_data.py               Market data module
    ├── atoms/
    │   └── alpaca_api/                  Alpaca API atoms
    └── molecules/
        └── alpaca_molecules/            Alpaca business logic
```

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (http://localhost/~wilsonb/)                       │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──(HTTP)──> Apache Server :80
             │            └──> ~/public_html/index.html
             │
             ├──(HTTP)──> Apache CGI :80
             │            └──> ~/public_html/cgi-bin/api/*.py
             │                 └──> Alpaca REST API
             │
             └──(WebSocket)──> Trade Stream Server :8765
                               └──> Alpaca WebSocket API
```

## Files Updated/Created

### Repository Files:
- ✅ `public_html/index.html` - Updated with WebSocket integration
- ✅ `services/trade_stream_server.py` - New WebSocket server
- ✅ `services/start_trade_stream.sh` - Startup script
- ✅ `services/market_sentinel_trade_stream.service` - Systemd service

### Apache Files:
- ✅ `~/public_html` - Symlink created
- ✅ `~/public_html/cgi-bin` - Symlink created

## How to Start Everything

### 1. Start WebSocket Server:
```bash
cd ~/dl/github.com/alpaca
./services/start_trade_stream.sh
```

### 2. Access Market Sentinel:
Open your browser to:
```
http://localhost/~wilsonb/
```

### 3. Test CGI API (optional):
```bash
curl "http://localhost/~wilsonb/cgi-bin/api/market_data_api.py?action=quote&symbol=AAPL"
```

## Troubleshooting

### Issue: "403 Forbidden" when accessing pages
**Solution:** Check file permissions
```bash
chmod 755 ~/dl/github.com/alpaca/public_html
chmod 644 ~/dl/github.com/alpaca/public_html/index.html
```

### Issue: "Internal Server Error" for CGI scripts
**Solution:** Check CGI script permissions and shebang
```bash
chmod 755 ~/dl/github.com/alpaca/cgi-bin/api/*.py
head -1 ~/dl/github.com/alpaca/cgi-bin/api/market_data_api.py
# Should show: #!/usr/bin/env python3
```

### Issue: "Module not found" errors in CGI
**Solution:** CGI scripts need to use absolute Python path
```bash
# Option 1: Use conda environment directly in shebang
#!/home/wilsonb/miniconda3/envs/alpaca/bin/python

# Option 2: Add to script's sys.path
```

### Issue: WebSocket connection fails
**Solution:** Make sure trade stream server is running
```bash
ps aux | grep trade_stream_server
./services/start_trade_stream.sh
```

## Apache Logs

View Apache error logs:
```bash
sudo tail -f /var/log/apache2/error.log
```

View Apache access logs:
```bash
sudo tail -f /var/log/apache2/access.log
```

## Next Steps

1. ✅ Create `.env` file with Alpaca credentials (if not already done)
2. ✅ Start the WebSocket server: `./services/start_trade_stream.sh`
3. ✅ Open browser to: `http://localhost/~wilsonb/`
4. ✅ Search for a symbol (e.g., AAPL, SPY, TSLA)
5. ✅ Watch real-time trades stream in!

---

**Status:** ✅ Apache server setup complete and ready to use!
