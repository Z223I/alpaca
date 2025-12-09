#!/usr/bin/env python3
"""
Consolidate trades from Webull CSV export.

This script reads a trade CSV file and consolidates buy/sell orders for each symbol,
handling various scenarios: 1:1, many:1, 1:many, and many:many relationships.

Usage:
    python molecules/consolidate_trades.py --input trades/webull_20251209.csv --output consolidated.csv
"""

import csv
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple


class Trade:
    """Represents a single trade from the CSV."""

    def __init__(self, row: Dict[str, str]):
        self.symbol = row['Symbol']
        self.side = row['Side']
        self.qty = int(row['Filled Qty'])
        self.price = float(row['Average Price'])
        # Parse datetime (ignore timezone for simplicity)
        filled_time_raw = row['Filled Time'].strip()
        if filled_time_raw:
            filled_time_str = filled_time_raw.rsplit(' ', 1)[0]  # Remove timezone
            self.filled_time = datetime.strptime(filled_time_str, '%m/%d/%Y %H:%M:%S')
        else:
            # If no filled time, use a default timestamp
            self.filled_time = datetime.min
        self.order_type = row['Order Type']
        self.limit_price = row['Limit Price']
        self.stop_price = row['Stop Price']

    def __repr__(self):
        return f"{self.side} {self.qty} {self.symbol} @ {self.price} on {self.filled_time}"


class ConsolidatedTrade:
    """Represents a consolidated trade pair or group."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buy_trades: List[Trade] = []
        self.sell_trades: List[Trade] = []

    def add_buy(self, trade: Trade):
        """Add a buy trade."""
        self.buy_trades.append(trade)

    def add_sell(self, trade: Trade):
        """Add a sell trade."""
        self.sell_trades.append(trade)

    def calculate(self) -> Dict[str, any]:
        """Calculate consolidated metrics."""
        if not self.buy_trades and not self.sell_trades:
            return None

        # Calculate buy side
        total_buy_qty = sum(t.qty for t in self.buy_trades)
        total_buy_cost = sum(t.qty * t.price for t in self.buy_trades)
        avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0

        # Calculate sell side
        total_sell_qty = sum(t.qty for t in self.sell_trades)
        total_sell_proceeds = sum(t.qty * t.price for t in self.sell_trades)
        avg_sell_price = total_sell_proceeds / total_sell_qty if total_sell_qty > 0 else 0

        # Calculate P&L on matched quantity
        matched_qty = min(total_buy_qty, total_sell_qty)
        realized_pnl = 0
        if matched_qty > 0:
            realized_pnl = (avg_sell_price - avg_buy_price) * matched_qty

        # Net position
        net_position = total_buy_qty - total_sell_qty

        # Date range
        all_trades = self.buy_trades + self.sell_trades
        first_trade_date = min(t.filled_time for t in all_trades) if all_trades else None
        last_trade_date = max(t.filled_time for t in all_trades) if all_trades else None

        return {
            'symbol': self.symbol,
            'total_buy_qty': total_buy_qty,
            'avg_buy_price': avg_buy_price,
            'total_buy_cost': total_buy_cost,
            'total_sell_qty': total_sell_qty,
            'avg_sell_price': avg_sell_price,
            'total_sell_proceeds': total_sell_proceeds,
            'matched_qty': matched_qty,
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': (realized_pnl / total_buy_cost * 100) if total_buy_cost > 0 else 0,
            'net_position': net_position,
            'num_buy_trades': len(self.buy_trades),
            'num_sell_trades': len(self.sell_trades),
            'first_trade_date': first_trade_date.strftime('%m/%d/%Y %H:%M:%S') if first_trade_date else '',
            'last_trade_date': last_trade_date.strftime('%m/%d/%Y %H:%M:%S') if last_trade_date else '',
        }


def read_trades(input_file: str) -> List[Trade]:
    """Read trades from CSV file."""
    trades = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(Trade(row))
    return trades


def consolidate_trades(trades: List[Trade]) -> List[Dict[str, any]]:
    """
    Consolidate trades by symbol.
    Groups all buys and sells for each symbol and calculates consolidated metrics.
    """
    # Group trades by symbol
    symbol_trades = defaultdict(lambda: ConsolidatedTrade(symbol=''))

    for trade in trades:
        if trade.symbol not in symbol_trades:
            symbol_trades[trade.symbol] = ConsolidatedTrade(trade.symbol)

        if trade.side == 'Buy':
            symbol_trades[trade.symbol].add_buy(trade)
        elif trade.side == 'Sell':
            symbol_trades[trade.symbol].add_sell(trade)

    # Calculate consolidated metrics for each symbol
    consolidated = []
    for symbol, consolidated_trade in symbol_trades.items():
        result = consolidated_trade.calculate()
        if result:
            consolidated.append(result)

    # Sort by last trade date (most recent first)
    consolidated.sort(key=lambda x: x['last_trade_date'], reverse=True)

    return consolidated


def write_consolidated_trades(output_file: str, consolidated_trades: List[Dict[str, any]]):
    """Write consolidated trades to CSV file."""
    if not consolidated_trades:
        print("No trades to write.")
        return

    fieldnames = [
        'symbol',
        'num_buy_trades',
        'total_buy_qty',
        'avg_buy_price',
        'total_buy_cost',
        'num_sell_trades',
        'total_sell_qty',
        'avg_sell_price',
        'total_sell_proceeds',
        'matched_qty',
        'realized_pnl',
        'realized_pnl_pct',
        'net_position',
        'first_trade_date',
        'last_trade_date',
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(consolidated_trades)

    print(f"Wrote {len(consolidated_trades)} consolidated trades to {output_file}")


def print_summary(consolidated_trades: List[Dict[str, any]]):
    """Print a summary of consolidated trades."""
    total_pnl = sum(t['realized_pnl'] for t in consolidated_trades)
    total_trades = len(consolidated_trades)
    winning_trades = sum(1 for t in consolidated_trades if t['realized_pnl'] > 0)
    losing_trades = sum(1 for t in consolidated_trades if t['realized_pnl'] < 0)

    print("\n" + "="*80)
    print("CONSOLIDATED TRADES SUMMARY")
    print("="*80)
    print(f"Total Symbols Traded: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {winning_trades/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
    print(f"Total Realized P&L: ${total_pnl:,.2f}")
    print("="*80)

    # Print top winners and losers
    sorted_by_pnl = sorted(consolidated_trades, key=lambda x: x['realized_pnl'], reverse=True)

    print("\nTop 5 Winners:")
    print(f"{'Symbol':<8} {'Buy Qty':<8} {'Avg Buy':<10} {'Sell Qty':<8} {'Avg Sell':<10} {'P&L':<12} {'P&L %':<8}")
    print("-"*80)
    for trade in sorted_by_pnl[:5]:
        print(f"{trade['symbol']:<8} {trade['total_buy_qty']:<8} "
              f"${trade['avg_buy_price']:<9.2f} {trade['total_sell_qty']:<8} "
              f"${trade['avg_sell_price']:<9.2f} ${trade['realized_pnl']:<11.2f} "
              f"{trade['realized_pnl_pct']:>6.2f}%")

    print("\nTop 5 Losers:")
    print(f"{'Symbol':<8} {'Buy Qty':<8} {'Avg Buy':<10} {'Sell Qty':<8} {'Avg Sell':<10} {'P&L':<12} {'P&L %':<8}")
    print("-"*80)
    for trade in sorted_by_pnl[-5:]:
        print(f"{trade['symbol']:<8} {trade['total_buy_qty']:<8} "
              f"${trade['avg_buy_price']:<9.2f} {trade['total_sell_qty']:<8} "
              f"${trade['avg_sell_price']:<9.2f} ${trade['realized_pnl']:<11.2f} "
              f"{trade['realized_pnl_pct']:>6.2f}%")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Consolidate trades from Webull CSV export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Consolidate trades and save to file
  python molecules/consolidate_trades.py --input trades/webull_20251209.csv --output consolidated.csv

  # Just view summary without saving
  python molecules/consolidate_trades.py --input trades/webull_20251209.csv
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path (Webull trade export)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output CSV file path for consolidated trades (optional)'
    )

    args = parser.parse_args()

    # Read trades
    print(f"Reading trades from {args.input}...")
    trades = read_trades(args.input)
    print(f"Read {len(trades)} trades")

    # Consolidate
    print("Consolidating trades...")
    consolidated_trades = consolidate_trades(trades)

    # Print summary
    print_summary(consolidated_trades)

    # Write output if specified
    if args.output:
        write_consolidated_trades(args.output, consolidated_trades)
        print(f"\nConsolidated trades written to: {args.output}")
    else:
        print("\nNo output file specified. Use --output to save consolidated trades.")


if __name__ == '__main__':
    main()
