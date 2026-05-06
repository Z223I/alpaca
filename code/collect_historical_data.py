#!/usr/bin/env python3
"""
Collect Historical Data

Retrieves 2-minute candlestick data from Alpaca for NASDAQ 100 symbols (or custom list)
over a date range. Data covers 04:00–10:30 ET per trading day. Saves one CSV per
symbol per day to ./historical_data_20/YYYY-MM-DD/{symbol}.csv.

Usage:
    python3 code/collect_historical_data.py
    python3 code/collect_historical_data.py --symbols AMD GOOGL AAPL
    python3 code/collect_historical_data.py --start-date 2025-03-01 --stop-date 2025-03-10
    python3 code/collect_historical_data.py --symbols AMD --display
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python-holidays'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import holidays
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from atoms.api.init_alpaca_client import init_alpaca_client

ET = ZoneInfo("America/New_York")
DATA_MASTER_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_master", "nasdaq_100.csv")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "historical_data_20")
SESSION_START = "04:00"
SESSION_END = "10:30"

CHART_WIDTH = 60   # characters wide for the price chart
CHART_HEIGHT = 12  # rows tall for the price chart


def load_nasdaq_100_symbols() -> list[str]:
    df = pd.read_csv(DATA_MASTER_CSV)
    return df["symbol"].dropna().str.strip().tolist()


def get_trading_days(start: date, stop: date) -> list[date]:
    nyse_holidays = holidays.NYSE(years=range(start.year, stop.year + 1))
    days = []
    current = start
    while current <= stop:
        if current.weekday() < 5 and current not in nyse_holidays:
            days.append(current)
        current += timedelta(days=1)
    return days


def fetch_bars(api: tradeapi.REST, symbol: str, trading_day: date) -> pd.DataFrame:
    start_et = datetime(trading_day.year, trading_day.month, trading_day.day, 4, 0, 0, tzinfo=ET)
    end_et = datetime(trading_day.year, trading_day.month, trading_day.day, 10, 30, 0, tzinfo=ET)

    bars = api.get_bars(
        symbol,
        TimeFrame(2, TimeFrame.Minute),
        start=start_et.isoformat(),
        end=end_et.isoformat(),
        adjustment="all",
        feed="sip",
    ).df

    if bars.empty:
        return bars

    bars.index = bars.index.tz_convert(ET)
    bars.index.name = "timestamp"
    return bars


def save_bars(bars: pd.DataFrame, symbol: str, trading_day: date):
    day_str = trading_day.strftime("%Y-%m-%d")
    out_dir = os.path.join(OUTPUT_ROOT, day_str)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}.csv")
    bars.to_csv(out_path)
    return out_path


def display_candlesticks(bars: pd.DataFrame, symbol: str, day_str: str):
    """Render an ASCII candlestick chart to stdout."""
    if bars.empty:
        print(f"  (no data)\n")
        return

    price_min = bars["low"].min()
    price_max = bars["high"].max()
    price_range = price_max - price_min or 1.0

    n = len(bars)
    col_width = max(1, CHART_WIDTH // n)

    # Build grid: rows=height, cols=n candles
    grid = [[" " * col_width for _ in range(n)] for _ in range(CHART_HEIGHT)]

    def price_to_row(p):
        # row 0 = top (high price), row CHART_HEIGHT-1 = bottom (low price)
        frac = (p - price_min) / price_range
        return int((1.0 - frac) * (CHART_HEIGHT - 1))

    for col_idx, (ts, row) in enumerate(bars.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        up = c >= o
        body_char = "█" if up else "░"
        wick_char = "│"

        row_h = price_to_row(h)
        row_l = price_to_row(l)
        row_o = price_to_row(max(o, c))
        row_c = price_to_row(min(o, c))

        for r in range(CHART_HEIGHT):
            cell = list(" " * col_width)
            mid = col_width // 2
            if row_h <= r <= row_l:
                if row_o <= r <= row_c:
                    # body
                    cell = list(body_char * col_width)
                else:
                    # wick in center column
                    cell[mid] = wick_char
            grid[r][col_idx] = "".join(cell)

    # Price labels on right axis
    print(f"\n  {symbol}  {day_str}  ({n} bars,  {price_min:.2f}–{price_max:.2f})")
    for r in range(CHART_HEIGHT):
        price_at_row = price_max - (r / (CHART_HEIGHT - 1)) * price_range
        label = f"{price_at_row:8.2f} │"
        print(label + "".join(grid[r]))

    # Time axis labels
    time_labels = [ts.strftime("%H:%M") for ts in bars.index]
    axis_line = " " * 10 + "└" + "─" * (n * col_width)
    print(axis_line)
    # Print a few evenly-spaced time labels
    tick_positions = list(range(0, n, max(1, n // 8)))
    label_row = " " * 11
    prev_end = 0
    for pos in tick_positions:
        char_pos = pos * col_width
        pad = char_pos - prev_end
        if pad >= 0:
            label_row += " " * pad + time_labels[pos]
            prev_end = char_pos + len(time_labels[pos])
    print(label_row)

    # Table: one row per bar
    print()
    print(f"  {'Time':>5}  {'Open':>8}  {'High':>8}  {'Low':>8}  {'Close':>8}  {'Vol':>10}  Dir")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  ───")
    for ts, row in bars.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        vol = int(row["volume"])
        direction = "▲" if c >= o else "▼"
        pct = (c - o) / o * 100 if o else 0
        print(f"  {ts.strftime('%H:%M'):>5}  {o:8.2f}  {h:8.2f}  {l:8.2f}  {c:8.2f}  {vol:>10,}  {direction} {pct:+.2f}%")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect 2-min historical candlestick data from Alpaca")
    parser.add_argument("--symbols", nargs="+", metavar="SYMBOL",
                        help="Symbols to collect (default: all NASDAQ 100 symbols)")
    parser.add_argument("--start-date", default="2026-02-01",
                        help="Start date YYYY-MM-DD (default: 2026-02-01)")
    parser.add_argument("--stop-date", default=date.today().strftime("%Y-%m-%d"),
                        help="Stop date YYYY-MM-DD (default: today)")
    parser.add_argument("--display", action="store_true",
                        help="Print ASCII candlestick chart and table for each file to stdout")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    symbols = args.symbols if args.symbols else load_nasdaq_100_symbols()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    stop = datetime.strptime(args.stop_date, "%Y-%m-%d").date()

    trading_days = get_trading_days(start, stop)
    print(f"Symbols: {len(symbols)}  |  Trading days: {len(trading_days)}  ({start} → {stop})")

    api = init_alpaca_client()

    skipped = 0
    saved = 0
    errors = 0

    for symbol in symbols:
        for day in trading_days:
            day_str = day.strftime("%Y-%m-%d")
            out_path = os.path.join(OUTPUT_ROOT, day_str, f"{symbol}.csv")

            if os.path.exists(out_path):
                if args.verbose:
                    print(f"  [skip] {symbol} {day_str} already exists")
                if args.display:
                    bars = pd.read_csv(out_path, index_col="timestamp", parse_dates=True)
                    bars.index = pd.DatetimeIndex(bars.index).tz_convert(ET)
                    display_candlesticks(bars, symbol, day_str)
                skipped += 1
                continue

            try:
                bars = fetch_bars(api, symbol, day)
                if bars.empty:
                    if args.verbose:
                        print(f"  [empty] {symbol} {day_str}")
                    skipped += 1
                    continue
                path = save_bars(bars, symbol, day)
                print(f"  [saved] {symbol} {day_str}  ({len(bars)} bars) → {path}")
                saved += 1
                if args.display:
                    display_candlesticks(bars, symbol, day_str)
            except Exception as e:
                print(f"  [error] {symbol} {day_str}: {e}")
                errors += 1

    print(f"\nDone. Saved={saved}  Skipped={skipped}  Errors={errors}")


if __name__ == "__main__":
    main()
