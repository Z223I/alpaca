#!/usr/bin/env python3
"""
Yahoo Finance Stock Screener
Filters stocks by float, price, and other criteria using Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import csv
import time
from typing import Dict, List, Optional
import argparse


class StockScreener:
    def __init__(self, input_file: str = "stocks_unfiltered.csv"):
        """Initialize the stock screener with input CSV file"""
        self.input_file = input_file
        self.stocks_data = []
        
    def load_symbols(self) -> List[str]:
        """Load stock symbols from CSV file"""
        symbols = []
        try:
            with open(self.input_file, 'r') as f:
                reader = csv.DictReader(f)
                symbols = [row['symbol'].strip() for row in reader]
            print(f"Loaded {len(symbols)} symbols from {self.input_file}")
            return symbols
        except FileNotFoundError:
            print(f"Error: {self.input_file} not found")
            return []
        except Exception as e:
            print(f"Error loading symbols: {e}")
            return []
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get stock information from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                return None
                
            # Get shares outstanding and float
            shares_outstanding = info.get('sharesOutstanding')
            float_shares = info.get('floatShares') or info.get('impliedSharesOutstanding')
            
            # Calculate market cap if not available
            market_cap = info.get('marketCap')
            if not market_cap and shares_outstanding and current_price:
                market_cap = shares_outstanding * current_price
            
            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'current_price': current_price,
                'market_cap': market_cap,
                'shares_outstanding': shares_outstanding,
                'float_shares': float_shares,
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume'),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'pe_ratio': info.get('trailingPE'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta')
            }
            
            return stock_data
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def meets_criteria(self, stock_data: Dict, 
                      min_price: float = 0.10, 
                      max_price: float = 25.00,
                      min_float: float = None,
                      max_float: float = None,
                      min_volume: int = None) -> bool:
        """Check if stock meets filtering criteria"""
        
        # Price filter
        price = stock_data.get('current_price')
        if not price or price < min_price or price > max_price:
            return False
        
        # Float filters (if specified)
        float_shares = stock_data.get('float_shares')
        if min_float and (not float_shares or float_shares < min_float):
            return False
        if max_float and (not float_shares or float_shares > max_float):
            return False
        
        # Volume filter (if specified)
        if min_volume:
            volume = stock_data.get('avg_volume')
            if not volume or volume < min_volume:
                return False
        
        return True
    
    def screen_stocks(self, 
                     min_price: float = 0.10,
                     max_price: float = 25.00,
                     min_float: float = None,
                     max_float: float = None,
                     min_volume: int = None,
                     delay: float = 0.1) -> List[Dict]:
        """Screen stocks based on criteria"""
        
        symbols = self.load_symbols()
        if not symbols:
            return []
        
        filtered_stocks = []
        total = len(symbols)
        
        print(f"\nScreening {total} stocks...")
        print(f"Criteria: Price ${min_price:.2f} - ${max_price:.2f}")
        if min_float:
            print(f"Min Float: {min_float:,.0f}")
        if max_float:
            print(f"Max Float: {max_float:,.0f}")
        if min_volume:
            print(f"Min Volume: {min_volume:,.0f}")
        print("-" * 50)
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Processing {symbol} ({i}/{total})...", end=' ')
            
            stock_data = self.get_stock_info(symbol)
            if stock_data and self.meets_criteria(stock_data, min_price, max_price, min_float, max_float, min_volume):
                filtered_stocks.append(stock_data)
                print(f"✓ PASS - Price: ${stock_data['current_price']:.2f}")
            else:
                reason = "No data"
                if stock_data:
                    price = stock_data.get('current_price', 0)
                    float_shares = stock_data.get('float_shares', 0)
                    if price < min_price or price > max_price:
                        reason = f"Price: ${price:.2f}"
                    elif min_float and float_shares and float_shares < min_float:
                        reason = f"Float: {float_shares:,.0f} (too low)"
                    elif max_float and float_shares and float_shares > max_float:
                        reason = f"Float: {float_shares:,.0f} (too high)"
                print(f"✗ FAIL - {reason}")
            
            # Rate limiting
            time.sleep(delay)
        
        return filtered_stocks
    
    def save_results(self, filtered_stocks: List[Dict], output_file: str = "stocks_filtered.csv"):
        """Save filtered results to CSV"""
        if not filtered_stocks:
            print("No stocks met the criteria")
            return
        
        df = pd.DataFrame(filtered_stocks)
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(filtered_stocks)} filtered stocks to {output_file}")
    
    def print_summary(self, filtered_stocks: List[Dict]):
        """Print summary of filtered stocks"""
        if not filtered_stocks:
            print("No stocks found matching criteria")
            return
        
        print(f"\n{'Symbol':<10} {'Price':<10} {'Float (M)':<12} {'Market Cap':<15} {'Name'}")
        print("-" * 70)
        
        for stock in filtered_stocks:
            float_m = stock.get('float_shares', 0) / 1_000_000 if stock.get('float_shares') else 0
            market_cap = stock.get('market_cap', 0)
            market_cap_str = f"${market_cap/1_000_000:.0f}M" if market_cap else "N/A"
            
            print(f"{stock['symbol']:<10} ${stock['current_price']:<9.2f} {float_m:<12.1f} {market_cap_str:<15} {stock.get('name', '')[:30]}")


def main():
    parser = argparse.ArgumentParser(description='Screen stocks using Yahoo Finance API')
    parser.add_argument('--min-price', type=float, default=0.10, help='Minimum price (default: 0.10)')
    parser.add_argument('--max-price', type=float, default=25.00, help='Maximum price (default: 25.00)')
    parser.add_argument('--min-float', type=float, help='Minimum float shares (e.g., 10000000 for 10M)')
    parser.add_argument('--max-float', type=float, help='Maximum float shares (e.g., 50000000 for 50M)')
    parser.add_argument('--min-volume', type=int, help='Minimum average volume')
    parser.add_argument('--input', default='stocks_unfiltered.csv', help='Input CSV file')
    parser.add_argument('--output', default='stocks_filtered.csv', help='Output CSV file')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between requests (seconds)')
    parser.add_argument('--summary-only', action='store_true', help='Only show summary, do not save CSV')
    
    args = parser.parse_args()
    
    # Initialize screener
    screener = StockScreener(args.input)
    
    # Screen stocks
    filtered_stocks = screener.screen_stocks(
        min_price=args.min_price,
        max_price=args.max_price,
        min_float=args.min_float,
        max_float=args.max_float,
        min_volume=args.min_volume,
        delay=args.delay
    )
    
    # Show results
    screener.print_summary(filtered_stocks)
    
    # Save results
    if not args.summary_only:
        screener.save_results(filtered_stocks, args.output)


if __name__ == "__main__":
    main()