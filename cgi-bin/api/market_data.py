#!/usr/bin/env python3
"""
Market Sentinel Flask API

This Flask application provides REST API endpoints for the Market Sentinel web interface.
It connects the frontend to the Alpaca market data backend.

Endpoints:
    GET /api/quote/<symbol>                     - Get latest quote for symbol
    GET /api/chart/<symbol>                     - Get chart data (supports interval and range params)
    GET /api/trades/<symbol>                    - Get time & sales (trade) data
    GET /api/health                             - Health check endpoint

Example Usage:
    curl http://localhost:5000/api/quote/AAPL
    curl http://localhost:5000/api/chart/AAPL?interval=1m&range=1d
    curl http://localhost:5000/api/trades/AAPL?limit=100
"""

import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import pytz

# Add paths for importing from molecules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molecules.alpaca_molecules.market_data import AlpacaMarketData  # noqa: E402

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global market data client
# Using paper trading account by default
market_data_client = None


def get_market_data_client():
    """
    Get or initialize the market data client.

    Returns:
        AlpacaMarketData instance
    """
    global market_data_client
    if market_data_client is None:
        # Initialize with default settings (can be changed via environment variables)
        provider = os.getenv('ALPACA_PROVIDER', 'alpaca')
        account_name = os.getenv('ALPACA_ACCOUNT_NAME', 'Bruce')
        account = os.getenv('ALPACA_ACCOUNT', 'paper')

        market_data_client = AlpacaMarketData(provider, account_name, account)

    return market_data_client


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON response with status and timestamp
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Market Sentinel API',
        'timestamp': datetime.now(pytz.timezone('America/New_York')).isoformat()
    })


@app.route('/api/quote/<symbol>', methods=['GET'])
def get_quote(symbol):
    """
    Get the latest quote for a stock symbol.

    Args:
        symbol: Stock symbol (path parameter)

    Returns:
        JSON response with quote data:
        {
            "symbol": "AAPL",
            "bid_price": 150.25,
            "ask_price": 150.30,
            "mid_price": 150.275,
            "bid_size": 100,
            "ask_size": 200,
            "timestamp": "2025-10-30T10:30:00-04:00"
        }
    """
    try:
        symbol = symbol.upper()
        client = get_market_data_client()
        quote_data = client.get_latest_quote_data(symbol)

        if 'error' in quote_data:
            return jsonify({
                'error': quote_data['error'],
                'symbol': symbol
            }), 500

        return jsonify(quote_data)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/chart/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """
    Get chart data for a stock symbol.

    Args:
        symbol: Stock symbol (path parameter)
        interval: Candlestick interval (query param, default: '1m')
                 Options: '10s', '20s', '30s', '1m', '5m', '30m', '1h', '1d', '1w', '1mo'
        range: Display range (query param, default: '1d')
              Options: '1d', '2d', '5d', '1mo', '1y'

    Returns:
        JSON response with chart data:
        {
            "symbol": "AAPL",
            "interval": "1m",
            "range": "1d",
            "bars": [
                {
                    "timestamp": "2025-10-30T09:30:00-04:00",
                    "open": 150.00,
                    "high": 150.50,
                    "low": 149.75,
                    "close": 150.25,
                    "volume": 100000
                },
                ...
            ],
            "start_date": "2025-10-29T09:30:00-04:00",
            "end_date": "2025-10-30T16:00:00-04:00",
            "bar_count": 390
        }
    """
    try:
        symbol = symbol.upper()
        interval = request.args.get('interval', '1m')
        range_str = request.args.get('range', '1d')

        client = get_market_data_client()
        chart_data = client.get_chart_data(symbol, interval, range_str)

        if 'error' in chart_data:
            return jsonify({
                'error': chart_data['error'],
                'symbol': symbol,
                'interval': interval,
                'range': range_str
            }), 500

        return jsonify(chart_data)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol
        }), 500


@app.route('/api/trades/<symbol>', methods=['GET'])
def get_trades(symbol):
    """
    Get time & sales (trade) data for a stock symbol.

    Args:
        symbol: Stock symbol (path parameter)
        limit: Maximum number of trades to return (query param, default: 100)

    Returns:
        JSON response with trade data:
        {
            "symbol": "AAPL",
            "trades": [
                {
                    "timestamp": "2025-10-30T10:30:15-04:00",
                    "price": 150.25,
                    "size": 100,
                    "exchange": "Q"
                },
                ...
            ],
            "trade_count": 100
        }
    """
    try:
        symbol = symbol.upper()
        limit = int(request.args.get('limit', 100))

        # Validate limit
        if limit < 1 or limit > 1000:
            return jsonify({
                'error': 'Limit must be between 1 and 1000',
                'symbol': symbol
            }), 400

        client = get_market_data_client()
        trades = client.get_time_and_sales(symbol, limit=limit)

        return jsonify({
            'symbol': symbol,
            'trades': trades,
            'trade_count': len(trades)
        })

    except ValueError:
        return jsonify({
            'error': 'Invalid limit parameter (must be integer)',
            'symbol': symbol
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'symbol': symbol
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/health',
            '/api/quote/<symbol>',
            '/api/chart/<symbol>',
            '/api/trades/<symbol>'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


# CGI Handler
def application(environ, start_response):
    """
    WSGI application for CGI deployment.

    This function is called by the web server when running as CGI.
    """
    return app(environ, start_response)


if __name__ == '__main__':
    # Development server
    # In production, this will be served via CGI
    print("Starting Market Sentinel API server...")
    print("Available endpoints:")
    print("  GET /api/health")
    print("  GET /api/quote/<symbol>")
    print("  GET /api/chart/<symbol>?interval=1m&range=1d")
    print("  GET /api/trades/<symbol>?limit=100")
    print()

    # Run development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
