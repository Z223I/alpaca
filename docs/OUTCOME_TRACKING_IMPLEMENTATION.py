"""
Outcome Tracking Implementation for Squeeze Alerts
===================================================

This file contains the complete implementation code to add outcome tracking
to squeeze_alerts.py. Follow the integration instructions at the bottom.

IMPORTANT: All constants are configurable variables at the top of the class.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


# =============================================================================
# PART 1: CONFIGURATION CONSTANTS
# =============================================================================
# Add these constants to the SqueezeAlertsMonitor class (after existing constants)

class OutcomeTrackingConstants:
    """Configuration constants for outcome tracking - all in one place for easy tuning"""

    # ===== OUTCOME TRACKING CONFIGURATION =====
    # Enable/disable outcome tracking globally
    OUTCOME_TRACKING_ENABLED = True

    # Duration to track outcomes after squeeze detection (minutes)
    OUTCOME_TRACKING_DURATION_MINUTES = 10

    # Interval for recording price snapshots (minutes)
    OUTCOME_TRACKING_INTERVAL_MINUTES = 1

    # Time tolerance for interval recording (seconds)
    # If a trade occurs within this window of the target interval time, record it
    OUTCOME_INTERVAL_TOLERANCE_SECONDS = 30

    # Maximum concurrent followups to track (memory/performance limit)
    OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100

    # Stop loss threshold (percentage below entry price)
    OUTCOME_STOP_LOSS_PERCENT = 7.5

    # Target gain thresholds to track achievement (percentages above entry)
    OUTCOME_TARGET_THRESHOLDS = [5.0, 10.0, 15.0]

    # Derived: List of all intervals to track [1, 2, 3, ..., 10]
    @property
    def OUTCOME_TRACKING_INTERVALS(self):
        return list(range(
            self.OUTCOME_TRACKING_INTERVAL_MINUTES,
            self.OUTCOME_TRACKING_DURATION_MINUTES + 1,
            self.OUTCOME_TRACKING_INTERVAL_MINUTES
        ))


# =============================================================================
# PART 2: DATA STRUCTURES
# =============================================================================
# Add these to SqueezeAlertsMonitor.__init__() method

def add_outcome_tracking_data_structures(self):
    """
    Add these lines to __init__() after existing data structure initialization
    (around line 125, after self.latest_spy_timestamp)
    """
    # ===== OUTCOME TRACKING DATA STRUCTURES =====
    # Active followups: tracks outcomes for squeezes in progress
    # Key format: "AAPL_2025-12-12_152045" (symbol_date_time)
    self.active_followups: Dict[str, Dict[str, Any]] = {}

    # Cumulative volume since squeeze start (for each followup)
    self.followup_volume_tracking: Dict[str, int] = {}

    # Cumulative trades since squeeze start (for each followup)
    self.followup_trades_tracking: Dict[str, int] = {}


# =============================================================================
# PART 3: MAIN OUTCOME TRACKING METHODS
# =============================================================================

class OutcomeTrackingMethods:
    """All outcome tracking methods - add these to SqueezeAlertsMonitor class"""

    def _start_outcome_tracking(self, symbol: str, squeeze_timestamp: datetime,
                                squeeze_price: float, alert_filename: str) -> None:
        """
        Initialize outcome tracking for a squeeze alert.

        Called from _report_squeeze() after saving the alert JSON file.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            squeeze_timestamp: Datetime when squeeze was detected
            squeeze_price: Price at squeeze detection (entry price for tracking)
            alert_filename: Name of the alert JSON file (to update later with outcomes)
        """
        if not self.OUTCOME_TRACKING_ENABLED:
            return

        # Check concurrent tracking limit
        if len(self.active_followups) >= self.OUTCOME_MAX_CONCURRENT_FOLLOWUPS:
            self.logger.warning(
                f"⚠️  Outcome tracking limit reached ({self.OUTCOME_MAX_CONCURRENT_FOLLOWUPS}). "
                f"Skipping tracking for {symbol}"
            )
            return

        # Create unique key for this followup: symbol_timestamp
        # Format: "AAPL_2025-12-12_152045"
        key = f"{symbol}_{squeeze_timestamp.strftime('%Y-%m-%d_%H%M%S')}"

        # Check if already tracking (shouldn't happen, but defensive)
        if key in self.active_followups:
            self.logger.warning(
                f"⚠️  Already tracking outcomes for {key}, skipping duplicate"
            )
            return

        # Calculate tracking window end time
        end_time = squeeze_timestamp + timedelta(
            minutes=self.OUTCOME_TRACKING_DURATION_MINUTES
        )

        # Calculate first interval time (T+1 minute)
        first_interval_time = squeeze_timestamp + timedelta(
            minutes=self.OUTCOME_TRACKING_INTERVAL_MINUTES
        )

        # Initialize target threshold tracking
        reached_targets = {}
        for threshold in self.OUTCOME_TARGET_THRESHOLDS:
            reached_targets[threshold] = {
                'reached': False,
                'minute': None,
                'price': None,
                'timestamp': None
            }

        # Initialize profitable snapshots for each interval
        profitable_snapshots = {
            i: None for i in range(1, self.OUTCOME_TRACKING_DURATION_MINUTES + 1)
        }

        # Create followup data structure
        self.active_followups[key] = {
            # Identification
            'symbol': symbol,
            'squeeze_timestamp': squeeze_timestamp,
            'squeeze_price': squeeze_price,
            'alert_filename': alert_filename,

            # Tracking window
            'start_time': squeeze_timestamp,
            'end_time': end_time,

            # Interval tracking state
            'next_interval': 1,
            'next_interval_time': first_interval_time,
            'intervals_recorded': [],
            'interval_data': {},

            # Running statistics (updated on every trade)
            'max_price_seen': squeeze_price,
            'min_price_seen': squeeze_price,
            'max_gain_percent': 0.0,
            'max_gain_minute': 0,
            'max_gain_timestamp': squeeze_timestamp,
            'max_drawdown_percent': 0.0,
            'max_drawdown_minute': 0,
            'max_drawdown_timestamp': squeeze_timestamp,

            # Stop loss tracking
            'reached_stop_loss': False,
            'stop_loss_minute': None,
            'stop_loss_price': None,
            'stop_loss_timestamp': None,

            # Target threshold tracking
            'reached_targets': reached_targets,

            # Profitability snapshots
            'profitable_snapshots': profitable_snapshots,

            # Last seen price (for handling gaps)
            'last_seen_price': squeeze_price,
            'last_seen_timestamp': squeeze_timestamp
        }

        # Initialize cumulative counters
        self.followup_volume_tracking[key] = 0
        self.followup_trades_tracking[key] = 0

        self.logger.info(
            f"📊 Started outcome tracking for {symbol} "
            f"(entry: ${squeeze_price:.4f}, duration: {self.OUTCOME_TRACKING_DURATION_MINUTES}min, "
            f"key: {key})"
        )


    def _check_outcome_intervals(self, symbol: str, timestamp: datetime,
                                  price: float, size: int) -> None:
        """
        Check if any outcome intervals are due for recording.

        Called from _handle_trade() on EVERY trade for symbols with active followups.
        Updates running statistics and records interval data when interval times are reached.

        Args:
            symbol: Stock symbol
            timestamp: Current trade timestamp
            price: Current trade price
            size: Current trade size
        """
        if not self.OUTCOME_TRACKING_ENABLED:
            return

        # Find all active followups for this symbol
        # Multiple squeezes for same symbol can be tracked concurrently
        keys_to_check = [k for k in self.active_followups.keys()
                        if k.startswith(f"{symbol}_")]

        if not keys_to_check:
            return  # No active tracking for this symbol

        keys_to_finalize = []

        for key in keys_to_check:
            followup = self.active_followups[key]

            # Update cumulative volume and trades
            self.followup_volume_tracking[key] += size
            self.followup_trades_tracking[key] += 1

            # Update last seen price (for gap handling)
            followup['last_seen_price'] = price
            followup['last_seen_timestamp'] = timestamp

            # Update running statistics (max/min, stop loss, targets)
            self._update_followup_statistics(key, price, timestamp)

            # Check if tracking period has ended
            if timestamp >= followup['end_time']:
                keys_to_finalize.append(key)
                continue

            # Check if market has closed (don't track into extended hours)
            if timestamp.time() >= datetime.strptime("16:00:00", "%H:%M:%S").time():
                self.logger.info(
                    f"🔔 Market closed, finalizing outcome tracking for {symbol} "
                    f"at {timestamp.strftime('%H:%M:%S')}"
                )
                keys_to_finalize.append(key)
                continue

            # Check if next interval is due
            next_interval_num = followup['next_interval']
            next_interval_time = followup['next_interval_time']

            # Define tolerance window
            tolerance = timedelta(seconds=self.OUTCOME_INTERVAL_TOLERANCE_SECONDS)

            # Check if current trade is within tolerance of next interval time
            if timestamp >= (next_interval_time - tolerance):
                # Record this interval
                self._record_outcome_interval(
                    key=key,
                    interval_num=next_interval_num,
                    timestamp=timestamp,
                    price=price,
                    volume=self.followup_volume_tracking[key],
                    trades=self.followup_trades_tracking[key]
                )

                # Advance to next interval
                next_interval_num += 1

                if next_interval_num <= self.OUTCOME_TRACKING_DURATION_MINUTES:
                    # More intervals to track
                    followup['next_interval'] = next_interval_num
                    followup['next_interval_time'] = followup['start_time'] + timedelta(
                        minutes=next_interval_num * self.OUTCOME_TRACKING_INTERVAL_MINUTES
                    )
                else:
                    # All intervals recorded, finalize
                    keys_to_finalize.append(key)

        # Finalize completed followups
        for key in keys_to_finalize:
            self._finalize_outcome_tracking(key)


    def _update_followup_statistics(self, key: str, price: float,
                                    timestamp: datetime) -> None:
        """
        Update running statistics for an active followup.

        Called on EVERY trade during the tracking period to capture:
        - Maximum price and gain (and when they occurred)
        - Minimum price and drawdown (and when they occurred)
        - Stop loss hits
        - Target threshold achievements

        This ensures we capture rapid moves that happen between interval snapshots.

        Args:
            key: Followup key (symbol_timestamp format)
            price: Current trade price
            timestamp: Current trade timestamp
        """
        followup = self.active_followups[key]
        squeeze_price = followup['squeeze_price']

        # Calculate gain/loss from squeeze entry price
        gain_percent = ((price - squeeze_price) / squeeze_price) * 100

        # Calculate elapsed time in minutes
        elapsed = (timestamp - followup['start_time']).total_seconds() / 60
        elapsed_minute = int(elapsed) + 1  # Convert to 1-indexed minute

        # Update maximum price and gain
        if price > followup['max_price_seen']:
            followup['max_price_seen'] = price
            followup['max_gain_percent'] = gain_percent
            followup['max_gain_minute'] = elapsed_minute
            followup['max_gain_timestamp'] = timestamp

            if self.verbose:
                self.logger.debug(
                    f"📈 {followup['symbol']} new high: ${price:.4f} "
                    f"({gain_percent:+.2f}%) at T+{elapsed_minute}min"
                )

        # Update minimum price and drawdown
        if price < followup['min_price_seen']:
            followup['min_price_seen'] = price
            followup['max_drawdown_percent'] = gain_percent
            followup['max_drawdown_minute'] = elapsed_minute
            followup['max_drawdown_timestamp'] = timestamp

            if self.verbose:
                self.logger.debug(
                    f"📉 {followup['symbol']} new low: ${price:.4f} "
                    f"({gain_percent:+.2f}%) at T+{elapsed_minute}min"
                )

        # Check stop loss threshold
        stop_loss_threshold = -self.OUTCOME_STOP_LOSS_PERCENT
        if not followup['reached_stop_loss'] and gain_percent <= stop_loss_threshold:
            followup['reached_stop_loss'] = True
            followup['stop_loss_price'] = price
            followup['stop_loss_minute'] = elapsed_minute
            followup['stop_loss_timestamp'] = timestamp

            self.logger.warning(
                f"🛑 {followup['symbol']} hit stop loss: ${price:.4f} "
                f"({gain_percent:.2f}%) at T+{elapsed_minute}min"
            )

        # Check target thresholds
        for threshold in self.OUTCOME_TARGET_THRESHOLDS:
            target_info = followup['reached_targets'][threshold]

            if not target_info['reached'] and gain_percent >= threshold:
                target_info['reached'] = True
                target_info['price'] = price
                target_info['minute'] = elapsed_minute
                target_info['timestamp'] = timestamp

                self.logger.info(
                    f"🎯 {followup['symbol']} hit +{threshold}% target: ${price:.4f} "
                    f"at T+{elapsed_minute}min"
                )


    def _record_outcome_interval(self, key: str, interval_num: int,
                                  timestamp: datetime, price: float,
                                  volume: int, trades: int) -> None:
        """
        Record data snapshot for a specific outcome interval.

        Called when a trade occurs at (or near) an interval time.
        Stores the price, volume, and other metrics at this point in time.

        Args:
            key: Followup key
            interval_num: Interval number (1 through OUTCOME_TRACKING_DURATION_MINUTES)
            timestamp: Trade timestamp (actual time, may differ slightly from target)
            price: Trade price at this interval
            volume: Cumulative volume since squeeze start
            trades: Cumulative number of trades since squeeze start
        """
        followup = self.active_followups[key]
        squeeze_price = followup['squeeze_price']

        # Calculate gain from entry
        gain_percent = ((price - squeeze_price) / squeeze_price) * 100

        # Record interval data
        followup['interval_data'][interval_num] = {
            'timestamp': timestamp.isoformat(),
            'price': float(price),
            'volume_since_squeeze': int(volume),
            'trades_since_squeeze': int(trades),
            'gain_percent': round(gain_percent, 2)
        }

        # Mark this interval as recorded
        followup['intervals_recorded'].append(interval_num)

        # Record profitability snapshot
        followup['profitable_snapshots'][interval_num] = (price > squeeze_price)

        if self.verbose:
            self.logger.debug(
                f"📊 {followup['symbol']} T+{interval_num}min: "
                f"${price:.4f} ({gain_percent:+.2f}%) "
                f"[{len(followup['intervals_recorded'])}/{self.OUTCOME_TRACKING_DURATION_MINUTES} intervals]"
            )


    def _build_outcome_summary(self, followup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build summary statistics from followup data.

        Called when outcome tracking is complete to calculate final metrics
        and aggregate statistics for analysis.

        Args:
            followup: Followup data dictionary

        Returns:
            Dictionary containing summary statistics
        """
        # Get final interval data (T+10 or last recorded interval)
        final_interval = self.OUTCOME_TRACKING_DURATION_MINUTES

        if final_interval in followup['interval_data']:
            final_data = followup['interval_data'][final_interval]
            final_price = final_data['price']
            final_gain = final_data['gain_percent']
        elif followup['intervals_recorded']:
            # Use last recorded interval if T+10 not reached
            last_interval = max(followup['intervals_recorded'])
            final_data = followup['interval_data'][last_interval]
            final_price = final_data['price']
            final_gain = final_data['gain_percent']
        else:
            # No intervals recorded (shouldn't happen, but defensive)
            final_price = followup['squeeze_price']
            final_gain = 0.0

        # Build target achievement summary
        targets_achieved = {}
        for threshold in self.OUTCOME_TARGET_THRESHOLDS:
            target_info = followup['reached_targets'][threshold]
            threshold_int = int(threshold)

            targets_achieved[f'achieved_{threshold_int}pct'] = target_info['reached']
            targets_achieved[f'time_to_{threshold_int}pct_minutes'] = target_info['minute']
            targets_achieved[f'price_at_{threshold_int}pct'] = (
                float(target_info['price']) if target_info['price'] is not None else None
            )

        # Build profitability summary for key intervals
        profitable_at = {}
        for interval in [1, 2, 5, 10]:
            if interval <= self.OUTCOME_TRACKING_DURATION_MINUTES:
                if interval in followup['profitable_snapshots']:
                    profitable_at[f'profitable_at_{interval}min'] = (
                        followup['profitable_snapshots'][interval]
                    )

        # Build complete summary
        summary = {
            # Max/min statistics
            'max_price': float(followup['max_price_seen']),
            'max_gain_percent': round(followup['max_gain_percent'], 2),
            'max_gain_reached_at_minute': followup['max_gain_minute'],

            'min_price': float(followup['min_price_seen']),
            'max_drawdown_percent': round(followup['max_drawdown_percent'], 2),
            'max_drawdown_reached_at_minute': followup['max_drawdown_minute'],

            # Final statistics
            'price_at_10min': float(final_price),
            'final_gain_percent': round(final_gain, 2),

            # Stop loss
            'reached_stop_loss': followup['reached_stop_loss'],
            'time_to_stop_loss_minutes': followup['stop_loss_minute'],
            'price_at_stop_loss': (
                float(followup['stop_loss_price'])
                if followup['stop_loss_price'] is not None else None
            ),

            # Profitability snapshots
            **profitable_at,

            # Target achievements
            **targets_achieved,

            # Tracking metadata
            'intervals_recorded': followup['intervals_recorded'],
            'intervals_recorded_count': len(followup['intervals_recorded']),
            'tracking_completed': (
                len(followup['intervals_recorded']) == self.OUTCOME_TRACKING_DURATION_MINUTES
            )
        }

        return summary


    def _finalize_outcome_tracking(self, key: str) -> None:
        """
        Finalize outcome tracking and save results.

        Called when:
        - All intervals have been recorded (10 minutes elapsed)
        - Market closes during tracking period
        - Tracking period end time is reached

        Calculates final summary statistics and updates the alert JSON file.

        Args:
            key: Followup key to finalize
        """
        if key not in self.active_followups:
            self.logger.warning(f"⚠️  Cannot finalize {key}: not found in active followups")
            return

        followup = self.active_followups[key]

        # Build outcome summary statistics
        summary = self._build_outcome_summary(followup)

        # Update alert JSON file with outcome data
        self._update_alert_with_outcomes(
            alert_filename=followup['alert_filename'],
            followup=followup,
            summary=summary
        )

        # Clean up tracking data
        del self.active_followups[key]

        if key in self.followup_volume_tracking:
            del self.followup_volume_tracking[key]

        if key in self.followup_trades_tracking:
            del self.followup_trades_tracking[key]

        # Log completion
        completion_status = "✅ COMPLETE" if summary['tracking_completed'] else "⏸️  PARTIAL"
        self.logger.info(
            f"{completion_status} Outcome tracking for {followup['symbol']}: "
            f"max gain {summary['max_gain_percent']:+.2f}% @ T+{summary['max_gain_reached_at_minute']}min, "
            f"final {summary['final_gain_percent']:+.2f}% @ T+10min "
            f"({summary['intervals_recorded_count']}/{self.OUTCOME_TRACKING_DURATION_MINUTES} intervals)"
        )


    def _update_alert_with_outcomes(self, alert_filename: str, followup: Dict[str, Any],
                                     summary: Dict[str, Any]) -> None:
        """
        Update the original alert JSON file with outcome tracking data.

        Reads the existing alert JSON, adds the outcome_tracking section,
        and writes it back to disk.

        Args:
            alert_filename: Name of alert JSON file (e.g., "alert_AAPL_2025-12-12_152045.json")
            followup: Complete followup data dictionary
            summary: Summary statistics dictionary
        """
        try:
            filepath = self.squeeze_alerts_sent_dir / alert_filename

            # Check if file exists
            if not filepath.exists():
                self.logger.error(
                    f"❌ Cannot update outcomes: alert file not found: {alert_filename}"
                )
                return

            # Read existing alert data
            with open(filepath, 'r') as f:
                alert_data = json.load(f)

            # Add outcome tracking section
            alert_data['outcome_tracking'] = {
                # Configuration
                'enabled': True,
                'tracking_start': followup['start_time'].isoformat(),
                'tracking_end': followup['end_time'].isoformat(),
                'squeeze_entry_price': float(followup['squeeze_price']),
                'duration_minutes': self.OUTCOME_TRACKING_DURATION_MINUTES,
                'interval_minutes': self.OUTCOME_TRACKING_INTERVAL_MINUTES,

                # Interval snapshots (every minute)
                'intervals': followup['interval_data'],

                # Summary statistics
                'summary': summary
            }

            # Write updated data back to file
            with open(filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)

            self.logger.debug(f"📝 Updated {alert_filename} with outcome data")

        except Exception as e:
            self.logger.error(f"❌ Error updating alert with outcomes: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# PART 4: INTEGRATION INSTRUCTIONS
# =============================================================================

"""
INTEGRATION INSTRUCTIONS
========================

Follow these steps to integrate outcome tracking into squeeze_alerts.py:

STEP 1: Add Configuration Constants
-----------------------------------
Location: After existing class constants (around line 70)

Add:
    # Copy all constants from OutcomeTrackingConstants class above
    OUTCOME_TRACKING_ENABLED = True
    OUTCOME_TRACKING_DURATION_MINUTES = 10
    OUTCOME_TRACKING_INTERVAL_MINUTES = 1
    # ... etc (all constants)


STEP 2: Add Data Structures
---------------------------
Location: In __init__() method, after line 125 (after self.latest_spy_timestamp)

Add:
    # ===== OUTCOME TRACKING DATA STRUCTURES =====
    self.active_followups: Dict[str, Dict[str, Any]] = {}
    self.followup_volume_tracking: Dict[str, int] = {}
    self.followup_trades_tracking: Dict[str, int] = {}


STEP 3: Add All Methods
-----------------------
Location: After _calculate_phase1_metrics() method (around line 1362)

Add all methods from OutcomeTrackingMethods class:
    - _start_outcome_tracking()
    - _check_outcome_intervals()
    - _update_followup_statistics()
    - _record_outcome_interval()
    - _build_outcome_summary()
    - _finalize_outcome_tracking()
    - _update_alert_with_outcomes()


STEP 4: Modify _save_squeeze_alert_sent()
-----------------------------------------
Location: Line 1822, method signature

CHANGE:
    def _save_squeeze_alert_sent(...) -> None:

TO:
    def _save_squeeze_alert_sent(...) -> str:

AND at the end of the method (before "except"), ADD:
    return filename


STEP 5: Modify _report_squeeze()
--------------------------------
Location: After line 1643 (after calling _save_squeeze_alert_sent)

CHANGE:
    self._save_squeeze_alert_sent(...)

TO:
    filename = self._save_squeeze_alert_sent(...)

    # Start outcome tracking for this squeeze
    self._start_outcome_tracking(
        symbol=symbol,
        squeeze_timestamp=timestamp,
        squeeze_price=last_price,
        alert_filename=filename
    )


STEP 6: Modify _handle_trade()
------------------------------
Location: After line 1051 (after SPY tracking, before _detect_squeeze)

ADD:
    # Check outcome tracking intervals for this symbol
    self._check_outcome_intervals(symbol, timestamp, price, size)


STEP 7: Add Import
-----------------
Location: Top of file with other imports

ENSURE these are imported:
    from typing import Dict, List, Optional, Any
    from datetime import datetime, timedelta


TESTING CHECKLIST
=================
After integration:

□ Syntax check: python3 -m py_compile cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
□ Service restart: sudo systemctl restart squeeze_alerts
□ Check logs: journalctl -u squeeze_alerts -f
□ Wait for squeeze alert
□ Verify "📊 Started outcome tracking" log message
□ Wait 10 minutes
□ Verify "✅ COMPLETE Outcome tracking" log message
□ Check alert JSON file has "outcome_tracking" section
□ Verify all 10 intervals recorded
□ Verify summary statistics populated

"""
