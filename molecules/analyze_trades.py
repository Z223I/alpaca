#!/usr/bin/env python3
"""
Analyze trading performance from Webull CSV export.

This script provides comprehensive analysis of trading performance including:
- Win rate and profit metrics
- Time-based performance analysis
- Symbol performance ranking
- Trade timing patterns
- Risk metrics

Usage:
    python molecules/analyze_trades.py --input trades/webull_20251209.csv --timeframe week
"""

import csv
import argparse
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import statistics
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


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
            self.filled_time = datetime.min

        self.order_type = row['Order Type']

    def __repr__(self):
        return f"{self.side} {self.qty} {self.symbol} @ {self.price} on {self.filled_time}"


class TradePosition:
    """Represents a complete trade position (buy + sell)."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buy_trades: List[Trade] = []
        self.sell_trades: List[Trade] = []
        self.entry_time = None
        self.exit_time = None
        self.pnl = 0
        self.pnl_pct = 0
        self.holding_time = None

    def add_buy(self, trade: Trade):
        self.buy_trades.append(trade)
        if self.entry_time is None or trade.filled_time < self.entry_time:
            self.entry_time = trade.filled_time

    def add_sell(self, trade: Trade):
        self.sell_trades.append(trade)
        if self.exit_time is None or trade.filled_time > self.exit_time:
            self.exit_time = trade.filled_time

    def calculate(self):
        """Calculate position metrics."""
        if not self.buy_trades or not self.sell_trades:
            return

        total_buy_qty = sum(t.qty for t in self.buy_trades)
        total_buy_cost = sum(t.qty * t.price for t in self.buy_trades)
        avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0

        total_sell_qty = sum(t.qty for t in self.sell_trades)
        total_sell_proceeds = sum(t.qty * t.price for t in self.sell_trades)
        avg_sell_price = total_sell_proceeds / total_sell_qty if total_sell_qty > 0 else 0

        matched_qty = min(total_buy_qty, total_sell_qty)
        self.pnl = (avg_sell_price - avg_buy_price) * matched_qty
        self.pnl_pct = (self.pnl / total_buy_cost * 100) if total_buy_cost > 0 else 0

        if self.entry_time and self.exit_time and self.entry_time != datetime.min and self.exit_time != datetime.min:
            self.holding_time = self.exit_time - self.entry_time

    def is_winner(self) -> bool:
        return self.pnl > 0

    def is_complete(self) -> bool:
        """Check if position has both buys and sells."""
        return len(self.buy_trades) > 0 and len(self.sell_trades) > 0


class PeriodStats:
    """Statistics for a specific time period."""

    def __init__(self, period_name: str, positions: List[TradePosition]):
        self.period_name = period_name
        self.positions = positions
        self.stats = self._calculate_stats()

    def _calculate_stats(self) -> Dict:
        """Calculate statistics for this period."""
        if not self.positions:
            return {
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
            }

        total_trades = len(self.positions)
        winners = [p for p in self.positions if p.is_winner()]
        losers = [p for p in self.positions if not p.is_winner()]

        total_pnl = sum(p.pnl for p in self.positions)
        gross_profit = sum(p.pnl for p in winners)
        gross_loss = abs(sum(p.pnl for p in losers))

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
        }


class TradeAnalyzer:
    """Analyzes trading performance."""

    def __init__(self, trades: List[Trade], timeframe: str = 'all'):
        self.all_trades = trades
        self.timeframe = timeframe
        self.positions: List[TradePosition] = []
        self.period_stats: List[PeriodStats] = []
        self._build_positions()
        self._group_by_period()

    def _build_positions(self):
        """Build complete positions from ALL trades."""
        # Group by symbol
        symbol_trades = defaultdict(lambda: TradePosition(symbol=''))

        for trade in self.all_trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = TradePosition(trade.symbol)

            if trade.side == 'Buy':
                symbol_trades[trade.symbol].add_buy(trade)
            elif trade.side == 'Sell':
                symbol_trades[trade.symbol].add_sell(trade)

        # Calculate metrics for each position
        for position in symbol_trades.values():
            position.calculate()
            if position.is_complete():
                self.positions.append(position)

    def _get_period_key(self, date: datetime) -> str:
        """Get the period key for grouping based on timeframe."""
        if self.timeframe == 'day':
            return date.strftime('%Y-%m-%d')
        elif self.timeframe == 'week':
            # ISO week: Monday is the first day
            iso_year, iso_week, _ = date.isocalendar()
            # Get the Monday of that week for display
            monday = datetime.strptime(f'{iso_year}-W{iso_week:02d}-1', '%Y-W%W-%w')
            return f"Week of {monday.strftime('%m/%d/%Y')}"
        elif self.timeframe == 'month':
            return date.strftime('%Y-%m (%B)')
        elif self.timeframe == 'year':
            return date.strftime('%Y')
        else:  # 'all'
            return 'All Time'

    def _group_by_period(self):
        """Group positions by time period."""
        if self.timeframe == 'all':
            # For 'all', treat everything as one period
            self.period_stats = [PeriodStats('All Time', self.positions)]
            return

        # Group positions by period
        period_groups = defaultdict(list)
        for position in self.positions:
            if position.exit_time and position.exit_time != datetime.min:
                period_key = self._get_period_key(position.exit_time)
                period_groups[period_key].append(position)

        # Sort periods chronologically
        sorted_periods = sorted(period_groups.keys())

        # Create PeriodStats for each period
        self.period_stats = [
            PeriodStats(period, period_groups[period])
            for period in sorted_periods
        ]

    def get_basic_stats(self) -> Dict:
        """Calculate basic trading statistics."""
        if not self.positions:
            return {}

        total_trades = len(self.positions)
        winners = [p for p in self.positions if p.is_winner()]
        losers = [p for p in self.positions if not p.is_winner()]

        total_pnl = sum(p.pnl for p in self.positions)
        gross_profit = sum(p.pnl for p in winners)
        gross_loss = abs(sum(p.pnl for p in losers))

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy: (Win% × Avg Win) - (Loss% × Avg Loss)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Largest win/loss
        largest_win = max((p.pnl for p in winners), default=0)
        largest_loss = min((p.pnl for p in losers), default=0)

        return {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
        }

    def get_symbol_performance(self) -> List[Dict]:
        """Analyze performance by symbol."""
        symbol_stats = defaultdict(lambda: {
            'trades': 0,
            'pnl': 0,
            'wins': 0,
            'losses': 0
        })

        for position in self.positions:
            stats = symbol_stats[position.symbol]
            stats['trades'] += 1
            stats['pnl'] += position.pnl
            if position.is_winner():
                stats['wins'] += 1
            else:
                stats['losses'] += 1

        # Convert to list and add win rate
        result = []
        for symbol, stats in symbol_stats.items():
            result.append({
                'symbol': symbol,
                'trades': stats['trades'],
                'pnl': stats['pnl'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            })

        # Sort by P&L
        result.sort(key=lambda x: x['pnl'], reverse=True)
        return result

    def get_time_patterns(self) -> Dict:
        """Analyze trading patterns by time."""
        hour_pnl = defaultdict(list)
        day_pnl = defaultdict(list)

        for position in self.positions:
            if position.entry_time and position.entry_time != datetime.min:
                hour = position.entry_time.hour
                day = position.entry_time.strftime('%A')
                hour_pnl[hour].append(position.pnl)
                day_pnl[day].append(position.pnl)

        # Calculate average P&L by hour
        hourly_stats = []
        for hour in sorted(hour_pnl.keys()):
            pnls = hour_pnl[hour]
            hourly_stats.append({
                'hour': hour,
                'trades': len(pnls),
                'avg_pnl': sum(pnls) / len(pnls),
                'total_pnl': sum(pnls)
            })

        # Calculate average P&L by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = []
        for day in day_order:
            if day in day_pnl:
                pnls = day_pnl[day]
                daily_stats.append({
                    'day': day,
                    'trades': len(pnls),
                    'avg_pnl': sum(pnls) / len(pnls),
                    'total_pnl': sum(pnls)
                })

        return {
            'hourly': hourly_stats,
            'daily': daily_stats
        }

    def get_holding_time_analysis(self) -> Dict:
        """Analyze performance by holding time."""
        positions_with_time = [p for p in self.positions if p.holding_time is not None]

        if not positions_with_time:
            return {}

        holding_times = [p.holding_time.total_seconds() / 60 for p in positions_with_time]  # minutes

        # Categorize by holding time
        categories = {
            'scalp (<5min)': [],
            'short (5-30min)': [],
            'medium (30min-2hr)': [],
            'long (>2hr)': []
        }

        for position in positions_with_time:
            minutes = position.holding_time.total_seconds() / 60
            if minutes < 5:
                categories['scalp (<5min)'].append(position)
            elif minutes < 30:
                categories['short (5-30min)'].append(position)
            elif minutes < 120:
                categories['medium (30min-2hr)'].append(position)
            else:
                categories['long (>2hr)'].append(position)

        result = {}
        for category, positions in categories.items():
            if positions:
                winners = [p for p in positions if p.is_winner()]
                result[category] = {
                    'trades': len(positions),
                    'win_rate': len(winners) / len(positions) * 100,
                    'avg_pnl': sum(p.pnl for p in positions) / len(positions),
                    'total_pnl': sum(p.pnl for p in positions)
                }

        return result

    def get_streak_analysis(self) -> Dict:
        """Analyze winning and losing streaks."""
        if not self.positions:
            return {}

        # Sort by exit time
        sorted_positions = sorted(
            [p for p in self.positions if p.exit_time and p.exit_time != datetime.min],
            key=lambda x: x.exit_time
        )

        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        streaks = []

        for position in sorted_positions:
            is_win = position.is_winner()

            if is_win:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    streaks.append(current_streak)
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

        if current_streak != 0:
            streaks.append(current_streak)

        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'current_streak': current_streak,
            'all_streaks': streaks
        }


def generate_plots(analyzer: TradeAnalyzer, timeframe: str, output_dir: str = './plots'):
    """Generate visualization plots."""
    # Create timestamped directory: ./plots/YYYYMMDD
    now = datetime.now()
    date_dir = now.strftime('%Y%m%d')
    timestamp = now.strftime('%H%M%S')
    full_output_dir = os.path.join(output_dir, date_dir)

    # Create output directory if it doesn't exist
    Path(full_output_dir).mkdir(parents=True, exist_ok=True)

    stats = analyzer.get_basic_stats()
    if not stats:
        print("No data to plot.")
        return

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Period-specific plots (when not 'all')
    if timeframe != 'all' and len(analyzer.period_stats) > 1:
        plot_period_pnl_comparison(analyzer, full_output_dir, timeframe, timestamp)
        plot_period_winrate_comparison(analyzer, full_output_dir, timeframe, timestamp)
        plot_period_metrics_comparison(analyzer, full_output_dir, timeframe, timestamp)

    # Standard plots (always generate)
    # 1. Cumulative P&L over time
    plot_cumulative_pnl(analyzer, full_output_dir, timeframe, timestamp)

    # 2. Performance by hour
    plot_hourly_performance(analyzer, full_output_dir, timeframe, timestamp)

    # 3. Performance by day of week
    plot_daily_performance(analyzer, full_output_dir, timeframe, timestamp)

    # 4. Holding time analysis
    plot_holding_time_analysis(analyzer, full_output_dir, timeframe, timestamp)

    # 5. Win/Loss distribution
    plot_winloss_distribution(analyzer, full_output_dir, timeframe, timestamp)

    # 6. Top symbols performance
    plot_symbol_performance(analyzer, full_output_dir, timeframe, timestamp)

    print(f"\nPlots saved to {full_output_dir}/")


def plot_period_pnl_comparison(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot P&L comparison across periods."""
    periods = [ps.period_name for ps in analyzer.period_stats]
    pnls = [ps.stats['total_pnl'] for ps in analyzer.period_stats]
    colors = ['green' if p >= 0 else 'red' for p in pnls]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(periods)), pnls, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'P&L by {timeframe.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{timeframe.capitalize()}', fontsize=12)
    ax.set_ylabel('P&L ($)', fontsize=12)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, pnl) in enumerate(zip(bars, pnls)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${pnl:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/period_pnl_comparison_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_period_winrate_comparison(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot win rate comparison across periods."""
    periods = [ps.period_name for ps in analyzer.period_stats]
    win_rates = [ps.stats['win_rate'] for ps in analyzer.period_stats]
    trade_counts = [ps.stats['total_trades'] for ps in analyzer.period_stats]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(periods)), win_rates, color='#2E86AB', alpha=0.7, edgecolor='black')

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Break-even (50%)')
    ax.set_title(f'Win Rate by {timeframe.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{timeframe.capitalize()}', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels on bars with trade count
    for i, (bar, wr, count) in enumerate(zip(bars, win_rates, trade_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{wr:.1f}%\n({count})',
                ha='center', va='bottom',
                fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/period_winrate_comparison_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_period_metrics_comparison(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot multiple metrics comparison across periods."""
    periods = [ps.period_name for ps in analyzer.period_stats]
    pnls = [ps.stats['total_pnl'] for ps in analyzer.period_stats]
    profit_factors = [min(ps.stats['profit_factor'], 10) for ps in analyzer.period_stats]  # Cap at 10 for display
    expectancies = [ps.stats['expectancy'] for ps in analyzer.period_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Profit Factor
    bars1 = ax1.bar(range(len(periods)), profit_factors, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Break-even (PF=1.0)')
    ax1.set_title(f'Profit Factor by {timeframe.capitalize()}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Profit Factor', fontsize=11)
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    for i, (bar, pf) in enumerate(zip(bars1, profit_factors)):
        height = bar.get_height()
        actual_pf = analyzer.period_stats[i].stats['profit_factor']
        label = f'{actual_pf:.2f}' if actual_pf < 999 else '∞'
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=8)

    # Expectancy
    colors2 = ['green' if e >= 0 else 'red' for e in expectancies]
    bars2 = ax2.bar(range(len(periods)), expectancies, color=colors2, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_title(f'Expectancy by {timeframe.capitalize()}', fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'{timeframe.capitalize()}', fontsize=11)
    ax2.set_ylabel('Expectancy ($)', fontsize=11)
    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(periods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (bar, exp) in enumerate(zip(bars2, expectancies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${exp:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=8)

    plt.suptitle(f'Performance Metrics by {timeframe.capitalize()}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/period_metrics_comparison_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_pnl(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot cumulative P&L over time."""
    sorted_positions = sorted(
        [p for p in analyzer.positions if p.exit_time and p.exit_time != datetime.min],
        key=lambda x: x.exit_time
    )

    if not sorted_positions:
        return

    dates = [p.exit_time for p in sorted_positions]
    cumulative_pnl = []
    running_total = 0

    for position in sorted_positions:
        running_total += position.pnl
        cumulative_pnl.append(running_total)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, cumulative_pnl, linewidth=2, color='#2E86AB', marker='o', markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(dates, cumulative_pnl, 0, where=[y >= 0 for y in cumulative_pnl],
                     alpha=0.3, color='green', label='Profit')
    ax.fill_between(dates, cumulative_pnl, 0, where=[y < 0 for y in cumulative_pnl],
                     alpha=0.3, color='red', label='Loss')

    ax.set_title(f'Cumulative P&L Over Time ({timeframe.upper()})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format x-axis dates
    if len(dates) > 20:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cumulative_pnl_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hourly_performance(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot performance by hour of day."""
    time_patterns = analyzer.get_time_patterns()
    hourly = time_patterns.get('hourly', [])

    if not hourly:
        return

    hours = [f"{h['hour'] % 12 or 12}{'AM' if h['hour'] < 12 else 'PM'}" for h in hourly]
    pnls = [h['total_pnl'] for h in hourly]
    colors = ['green' if p >= 0 else 'red' for p in pnls]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(hours, pnls, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'Performance by Hour of Day ({timeframe.upper()})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Total P&L ($)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hourly_performance_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_daily_performance(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot performance by day of week."""
    time_patterns = analyzer.get_time_patterns()
    daily = time_patterns.get('daily', [])

    if not daily:
        return

    days = [d['day'] for d in daily]
    pnls = [d['total_pnl'] for d in daily]
    colors = ['green' if p >= 0 else 'red' for p in pnls]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(days, pnls, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'Performance by Day of Week ({timeframe.upper()})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Total P&L ($)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_performance_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_holding_time_analysis(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot performance by holding time category."""
    holding_analysis = analyzer.get_holding_time_analysis()

    if not holding_analysis:
        return

    categories = list(holding_analysis.keys())
    win_rates = [holding_analysis[cat]['win_rate'] for cat in categories]
    total_pnls = [holding_analysis[cat]['total_pnl'] for cat in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Win rate by holding time
    ax1.bar(categories, win_rates, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_title('Win Rate by Holding Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    for i, v in enumerate(win_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

    # P&L by holding time
    colors = ['green' if p >= 0 else 'red' for p in total_pnls]
    ax2.bar(categories, total_pnls, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_title('Total P&L by Holding Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total P&L ($)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(total_pnls):
        ax2.text(i, v, f'${v:.2f}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Holding Time Category', fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.suptitle(f'Holding Time Analysis ({timeframe.upper()})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/holding_time_analysis_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_winloss_distribution(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot win/loss distribution."""
    wins = [p.pnl for p in analyzer.positions if p.is_winner()]
    losses = [p.pnl for p in analyzer.positions if not p.is_winner()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    all_pnls = [p.pnl for p in analyzer.positions]
    ax1.hist(all_pnls, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.set_title('P&L Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('P&L ($)', fontsize=11)
    ax1.set_ylabel('Number of Trades', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # Box plot
    data_to_plot = []
    labels = []
    if wins:
        data_to_plot.append(wins)
        labels.append(f'Wins\n(n={len(wins)})')
    if losses:
        data_to_plot.append(losses)
        labels.append(f'Losses\n(n={len(losses)})')

    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                         showmeans=True, meanline=True)
        colors = ['green', 'red'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_title('Win/Loss Box Plot', fontsize=12, fontweight='bold')
        ax2.set_ylabel('P&L ($)', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Win/Loss Distribution ({timeframe.upper()})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/winloss_distribution_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_symbol_performance(analyzer: TradeAnalyzer, output_dir: str, timeframe: str, timestamp: str):
    """Plot top symbols by performance."""
    symbol_perf = analyzer.get_symbol_performance()

    if not symbol_perf:
        return

    # Get top 10 by absolute P&L
    top_10 = sorted(symbol_perf, key=lambda x: abs(x['pnl']), reverse=True)[:10]
    symbols = [s['symbol'] for s in top_10]
    pnls = [s['pnl'] for s in top_10]
    colors = ['green' if p >= 0 else 'red' for p in pnls]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(symbols, pnls, color=colors, alpha=0.7, edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'Top 10 Symbols by P&L ({timeframe.upper()})', fontsize=14, fontweight='bold')
    ax.set_xlabel('P&L ($)', fontsize=12)
    ax.set_ylabel('Symbol', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, pnl) in enumerate(zip(bars, pnls)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' ${pnl:.2f}',
                ha='left' if pnl >= 0 else 'right',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/symbol_performance_{timeframe}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def read_trades(input_file: str) -> List[Trade]:
    """Read trades from CSV file."""
    trades = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(Trade(row))
    return trades


def print_analysis(analyzer: TradeAnalyzer, timeframe: str):
    """Print comprehensive analysis."""
    print("\n" + "="*80)
    print(f"TRADE ANALYSIS - {timeframe.upper()} TIMEFRAME")
    print("="*80)

    if not analyzer.period_stats:
        print("No complete trades found.")
        return

    # Period-by-period breakdown (for day, week, month, year)
    if timeframe != 'all' and len(analyzer.period_stats) > 1:
        print("\n" + "-"*80)
        print(f"PERIOD-BY-PERIOD BREAKDOWN ({len(analyzer.period_stats)} periods)")
        print("-"*80)
        print(f"{'Period':<20} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'PF':<8} {'Exp':<10}")
        print("-"*80)

        cumulative_pnl = 0
        for period_stat in analyzer.period_stats:
            stats = period_stat.stats
            cumulative_pnl += stats['total_pnl']
            pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 999 else "∞"

            print(f"{period_stat.period_name:<20} {stats['total_trades']:<8} "
                  f"{stats['win_rate']:>5.1f}%  ${stats['total_pnl']:>9.2f}  "
                  f"{pf_str:<8} ${stats['expectancy']:>7.2f}")

        print("-"*80)
        print(f"{'CUMULATIVE':<20} {'':<8} {'':<8} ${cumulative_pnl:>9.2f}")
        print("-"*80)

    # Overall aggregate stats
    stats = analyzer.get_basic_stats()
    if not stats:
        return

    print("\n" + "-"*80)
    print("OVERALL PERFORMANCE SUMMARY")
    print("-"*80)
    print(f"Total Trades:       {stats['total_trades']}")
    print(f"Winners:            {stats['winners']} ({stats['win_rate']:.1f}%)")
    print(f"Losers:             {stats['losers']}")
    print(f"\nTotal P&L:          ${stats['total_pnl']:,.2f}")
    print(f"Gross Profit:       ${stats['gross_profit']:,.2f}")
    print(f"Gross Loss:         ${stats['gross_loss']:,.2f}")
    print(f"\nAverage Win:        ${stats['avg_win']:,.2f}")
    print(f"Average Loss:       ${abs(stats['avg_loss']):,.2f}")
    print(f"Profit Factor:      {stats['profit_factor']:.2f}")
    print(f"Expectancy:         ${stats['expectancy']:,.2f} per trade")
    print(f"\nLargest Win:        ${stats['largest_win']:,.2f}")
    print(f"Largest Loss:       ${stats['largest_loss']:,.2f}")

    # Symbol performance
    print("\n" + "-"*80)
    print("TOP 10 SYMBOLS BY P&L")
    print("-"*80)
    print(f"{'Symbol':<8} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win Rate':<12} {'Total P&L'}")
    print("-"*80)

    symbol_perf = analyzer.get_symbol_performance()
    for symbol_stats in symbol_perf[:10]:
        print(f"{symbol_stats['symbol']:<8} {symbol_stats['trades']:<8} "
              f"{symbol_stats['wins']:<6} {symbol_stats['losses']:<8} "
              f"{symbol_stats['win_rate']:>6.1f}%      ${symbol_stats['pnl']:>8.2f}")

    # Time patterns
    time_patterns = analyzer.get_time_patterns()

    if time_patterns.get('hourly'):
        print("\n" + "-"*80)
        print("PERFORMANCE BY HOUR OF DAY")
        print("-"*80)
        print(f"{'Hour':<8} {'Trades':<8} {'Avg P&L':<12} {'Total P&L'}")
        print("-"*80)
        for hour_stat in time_patterns['hourly']:
            hour_12 = hour_stat['hour'] % 12
            if hour_12 == 0:
                hour_12 = 12
            am_pm = 'AM' if hour_stat['hour'] < 12 else 'PM'
            print(f"{hour_12:>2}:00{am_pm:<2} {hour_stat['trades']:<8} "
                  f"${hour_stat['avg_pnl']:>9.2f}   ${hour_stat['total_pnl']:>9.2f}")

    if time_patterns.get('daily'):
        print("\n" + "-"*80)
        print("PERFORMANCE BY DAY OF WEEK")
        print("-"*80)
        print(f"{'Day':<12} {'Trades':<8} {'Avg P&L':<12} {'Total P&L'}")
        print("-"*80)
        for day_stat in time_patterns['daily']:
            print(f"{day_stat['day']:<12} {day_stat['trades']:<8} "
                  f"${day_stat['avg_pnl']:>9.2f}   ${day_stat['total_pnl']:>9.2f}")

    # Holding time analysis
    holding_analysis = analyzer.get_holding_time_analysis()
    if holding_analysis:
        print("\n" + "-"*80)
        print("PERFORMANCE BY HOLDING TIME")
        print("-"*80)
        print(f"{'Category':<20} {'Trades':<8} {'Win Rate':<12} {'Avg P&L':<12} {'Total P&L'}")
        print("-"*80)
        for category, stats in holding_analysis.items():
            print(f"{category:<20} {stats['trades']:<8} {stats['win_rate']:>6.1f}%      "
                  f"${stats['avg_pnl']:>9.2f}   ${stats['total_pnl']:>9.2f}")

    # Streak analysis
    streaks = analyzer.get_streak_analysis()
    if streaks:
        print("\n" + "-"*80)
        print("STREAK ANALYSIS")
        print("-"*80)
        print(f"Max Winning Streak:  {streaks['max_win_streak']} trades")
        print(f"Max Losing Streak:   {streaks['max_loss_streak']} trades")
        current = streaks['current_streak']
        if current > 0:
            print(f"Current Streak:      {current} wins")
        elif current < 0:
            print(f"Current Streak:      {abs(current)} losses")
        else:
            print(f"Current Streak:      N/A")

    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze trading performance from Webull CSV export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all trades
  python molecules/analyze_trades.py --input trades/webull_20251209.csv

  # Analyze last week only with plots
  python molecules/analyze_trades.py --input trades/webull_20251209.csv --timeframe week --plots

  # Analyze today's trades and save plots to custom directory
  python molecules/analyze_trades.py --input trades/webull_20251209.csv --timeframe day --plots --output-dir ./my_plots
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path (Webull trade export)'
    )

    parser.add_argument(
        '--timeframe', '-t',
        choices=['day', 'week', 'month', 'year', 'all'],
        default='all',
        help='Timeframe for analysis (default: all)'
    )

    parser.add_argument(
        '--plots', '-p',
        action='store_true',
        help='Generate plots and save to ./plots directory'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='./plots',
        help='Output directory for plots (default: ./plots)'
    )

    args = parser.parse_args()

    # Read trades
    print(f"Reading trades from {args.input}...")
    trades = read_trades(args.input)
    print(f"Read {len(trades)} trades")

    # Analyze
    analyzer = TradeAnalyzer(trades, args.timeframe)
    print_analysis(analyzer, args.timeframe)

    # Generate plots if requested
    if args.plots:
        print("\nGenerating plots...")
        generate_plots(analyzer, args.timeframe, args.output_dir)


if __name__ == '__main__':
    main()
