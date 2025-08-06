import requests
import json
from datetime import datetime, timedelta
import os


class AlpacaDailyPnL:
    def __init__(self, api_key, secret_key,
                 base_url="https://paper-api.alpaca.markets"):
        """
        Initialize Alpaca API client

        Args:
            api_key: Your Alpaca API key
            secret_key: Your Alpaca secret key
            base_url: API base URL (use "https://api.alpaca.markets"
                     for live trading)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key,
            'Content-Type': 'application/json'
        }

    def get_account_info(self):
        """Get current account information"""
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            # Display configuration details on 403 errors to help with debugging
            if response.status_code == 403:
                print("\n" + "="*60)
                print("🔒 403 FORBIDDEN ERROR - Account Configuration Check")
                print("="*60)
                print(f"API Key:      {self.api_key}")
                print(f"Secret Key:   {self.secret_key[:8]}...{self.secret_key[-4:] if len(self.secret_key) > 12 else '***'}")
                print(f"Base URL:     {self.base_url}")
                print("\nPlease verify:")
                print("• API Key matches your account configuration")
                print("• Secret Key is correct and not expired")
                print("• Base URL matches your account type (paper vs live)")
                print("• Account has proper permissions")
                print("="*60)
            
            raise Exception(f"Error getting account info: "
                            f"{response.status_code} - {response.text}")

    def get_portfolio_history(self, period="1D", timeframe="1Min",
                              extended_hours=None):
        """
        Get portfolio history for calculating daily P&L

        Args:
            period: Time period (1D for 1 day, 1W for 1 week, etc.)
            timeframe: Data timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            extended_hours: Include extended hours trading (True/False)
        """
        url = f"{self.base_url}/v2/account/portfolio/history"
        params = {
            'period': period,
            'timeframe': timeframe
        }

        if extended_hours is not None:
            params['extended_hours'] = extended_hours

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error getting portfolio history: "
                            f"{response.status_code} - {response.text}")

    def calculate_daily_pnl(self):
        """Calculate current day profit and loss with percentage"""
        try:
            # Get account info for current equity
            account = self.get_account_info()
            current_equity = float(account['equity'])

            # Get portfolio history for the day
            portfolio_history = self.get_portfolio_history(
                period="1D", timeframe="1Min")

            # Extract equity values from portfolio history
            equity_values = portfolio_history.get('equity', [])

            if not equity_values or len(equity_values) < 2:
                print("Insufficient data to calculate daily P&L")
                return None

            # Get starting equity (first value of the day)
            starting_equity = float(equity_values[0])

            # Calculate P&L
            daily_pnl = current_equity - starting_equity
            daily_pnl_percentage = ((daily_pnl / starting_equity) * 100
                                    if starting_equity != 0 else 0)

            result = {
                'current_equity': current_equity,
                'starting_equity': starting_equity,
                'daily_pnl': daily_pnl,
                'daily_pnl_percentage': daily_pnl_percentage,
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            print(f"Error calculating daily P&L: {str(e)}")
            return None

    def get_positions(self):
        """Get current positions for additional context"""
        url = f"{self.base_url}/v2/positions"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error getting positions: "
                            f"{response.status_code} - {response.text}")

    def display_daily_summary(self):
        """Display a formatted daily P&L summary"""
        pnl_data = self.calculate_daily_pnl()

        if pnl_data:
            print("=" * 50)
            print("DAILY PROFIT & LOSS SUMMARY")
            print("=" * 50)
            print(f"Starting Equity: ${pnl_data['starting_equity']:,.2f}")
            print(f"Current Equity:  ${pnl_data['current_equity']:,.2f}")
            print(f"Daily P&L:       ${pnl_data['daily_pnl']:,.2f}")
            print(f"Daily P&L %:     {pnl_data['daily_pnl_percentage']:+.2f}%")
            print(f"Timestamp:       {pnl_data['timestamp']}")

            # Color coding for terminal output
            if pnl_data['daily_pnl'] > 0:
                status = "📈 PROFIT"
            elif pnl_data['daily_pnl'] < 0:
                status = "📉 LOSS"
            else:
                status = "➡️ BREAK EVEN"

            print(f"Status:          {status}")
            print("=" * 50)
        else:
            print("Unable to calculate daily P&L")

    def create_pnl(self):
        """Create PnL summary using instance credentials"""
        self.display_daily_summary()
