"""
Tests for MomentumAlertsSystem SIP delay fixes.

FILE: tests/test_momentum_alerts_sip_delay.py
METHODS UNDER TEST (code/momentum_alerts.py):
  - MomentumAlertsSystem._collect_stock_data
  - MomentumAlertsSystem._get_market_open_price

REMINDER: Run these tests whenever either method above is modified.
  conda run -n alpaca python -m pytest tests/test_momentum_alerts_sip_delay.py -v

Covers two changes introduced to make the system work with 30-minute
delayed SIP data on a free-tier Alpaca account:

1. _collect_stock_data  — window widened from 30 to 60 minutes so that
   there are always enough bars for EMA9 / momentum calculations,
   even early in the trading session.

2. _get_market_open_price — the SIP delay cap was removed from end_time
   because the 9:30 AM bar is fixed historical data, accessible at any
   time, and the cap caused start_time > end_time before 9:55 AM.
"""

import asyncio
import pytest
import pytz
from datetime import datetime, timedelta
from datetime import datetime as real_datetime
from unittest.mock import MagicMock, Mock, patch, call

from code.momentum_alerts import MomentumAlertsSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ET_TZ = pytz.timezone('US/Eastern')


def _make_system() -> MomentumAlertsSystem:
    """
    Return a MomentumAlertsSystem instance with all heavy dependencies
    mocked out so no filesystem I/O or network calls are made.
    """
    with patch('code.momentum_alerts.init_alpaca_client', return_value=Mock()), \
         patch('code.momentum_alerts.BreakoutDetector', return_value=Mock()), \
         patch('code.momentum_alerts.TelegramPoster', return_value=Mock()), \
         patch('code.momentum_alerts.UserManager', return_value=Mock()), \
         patch('code.momentum_alerts.get_momentum_alerts_config', return_value=Mock()), \
         patch('code.momentum_alerts.FundamentalDataFetcher', return_value=Mock()), \
         patch('pathlib.Path.mkdir'):
        system = MomentumAlertsSystem.__new__(MomentumAlertsSystem)

    system.et_tz = ET_TZ
    system.logger = Mock()
    system.historical_client = Mock()
    system.historical_client.get_bars.return_value = []
    return system


def _parse_utc(ts_str: str) -> datetime:
    """Parse a '%Y-%m-%dT%H:%M:%SZ' string into a UTC-aware datetime."""
    return real_datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.UTC)


def _make_bar(time_et: datetime, open_price: float) -> Mock:
    """Create a minimal mock bar with .t (UTC) and .o attributes."""
    bar = Mock()
    bar.t = time_et.astimezone(pytz.UTC)
    bar.o = open_price
    return bar


class FakeDatetime:
    """
    Drop-in replacement for the `datetime` class in code.momentum_alerts
    that overrides `now()` while delegating `combine()` and construction
    to the real datetime class.
    """

    _fake_now: datetime = None  # set per-test

    @classmethod
    def now(cls, tz=None):
        dt = cls._fake_now
        if tz is not None and dt.tzinfo is None:
            return tz.localize(dt)
        return dt

    @staticmethod
    def combine(*args, **kwargs):
        return real_datetime.combine(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        return real_datetime(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tests: _collect_stock_data window
# ---------------------------------------------------------------------------

class TestCollectStockDataWindow:
    """
    _collect_stock_data should fetch a 60-minute window ending at
    now - SIP_DELAY_MINUTES so that there are always enough bars for
    EMA9 and momentum calculations, even early in the trading session.
    """

    def test_window_is_60_minutes_wide(self):
        """start_time must be exactly 60 minutes before end_time."""
        system = _make_system()

        asyncio.run(system._collect_stock_data(['AAPL']))

        ca = system.historical_client.get_bars.call_args
        start_dt = _parse_utc(ca.kwargs['start'])
        end_dt = _parse_utc(ca.kwargs['end'])
        diff_minutes = (end_dt - start_dt).total_seconds() / 60

        assert diff_minutes == 60, (
            f"Expected 60-minute window, got {diff_minutes:.1f} minutes. "
            f"The window was widened from 30 to 60 min to ensure enough bars."
        )

    def test_end_time_is_sip_delayed(self):
        """end_time must be approximately now - SIP_DELAY_MINUTES (30 min)."""
        system = _make_system()

        before = real_datetime.now(pytz.UTC)
        asyncio.run(system._collect_stock_data(['AAPL']))
        after = real_datetime.now(pytz.UTC)

        ca = system.historical_client.get_bars.call_args
        end_dt = _parse_utc(ca.kwargs['end'])

        # Timestamps are truncated to seconds by strftime; allow ±1 s tolerance
        tolerance = timedelta(seconds=1)
        delay = timedelta(minutes=MomentumAlertsSystem.SIP_DELAY_MINUTES)
        assert (before - delay - tolerance) <= end_dt <= (after - delay + tolerance), (
            f"end_time {end_dt} not in expected range "
            f"[{before - delay}, {after - delay}]"
        )

    def test_start_time_is_90_minutes_before_now(self):
        """start_time must be approximately now - 90 min (delay + window)."""
        system = _make_system()

        before = real_datetime.now(pytz.UTC)
        asyncio.run(system._collect_stock_data(['AAPL']))
        after = real_datetime.now(pytz.UTC)

        ca = system.historical_client.get_bars.call_args
        start_dt = _parse_utc(ca.kwargs['start'])

        # Timestamps are truncated to seconds by strftime; allow ±1 s tolerance
        tolerance = timedelta(seconds=1)
        expected_offset = timedelta(
            minutes=MomentumAlertsSystem.SIP_DELAY_MINUTES + 60
        )
        assert (before - expected_offset - tolerance) <= start_dt <= (after - expected_offset + tolerance), (
            f"start_time {start_dt} not in expected 90-min-ago range"
        )

    def test_sip_feed_is_used(self):
        """get_bars must be called with feed='sip'."""
        system = _make_system()

        asyncio.run(system._collect_stock_data(['AAPL']))

        ca = system.historical_client.get_bars.call_args
        assert ca.kwargs.get('feed') == 'sip'

    def test_returns_empty_dict_no_client(self):
        """Returns {} when historical_client is None."""
        system = _make_system()
        system.historical_client = None

        result = asyncio.run(system._collect_stock_data(['AAPL']))

        assert result == {}

    def test_returns_empty_dict_no_symbols(self):
        """Returns {} immediately when symbols list is empty; no API calls."""
        system = _make_system()

        result = asyncio.run(system._collect_stock_data([]))

        assert result == {}
        system.historical_client.get_bars.assert_not_called()

    def test_returns_empty_dict_when_bars_empty(self):
        """Returns {} when the API returns no bars for the symbol."""
        system = _make_system()
        system.historical_client.get_bars.return_value = []

        result = asyncio.run(system._collect_stock_data(['AAPL']))

        assert result == {}

    def test_window_is_wider_than_old_30_min_window(self):
        """
        Regression: the window must be > 30 min to fix the early-session
        bar shortage bug introduced by the SIP delay change.
        """
        system = _make_system()

        asyncio.run(system._collect_stock_data(['AAPL']))

        ca = system.historical_client.get_bars.call_args
        start_dt = _parse_utc(ca.kwargs['start'])
        end_dt = _parse_utc(ca.kwargs['end'])
        diff_minutes = (end_dt - start_dt).total_seconds() / 60

        assert diff_minutes > 30, (
            f"Window ({diff_minutes:.0f} min) is not wider than the old 30-min window."
        )


# ---------------------------------------------------------------------------
# Tests: _get_market_open_price end_time cap removed
# ---------------------------------------------------------------------------

class TestGetMarketOpenPrice:
    """
    _get_market_open_price must use end_time = market_open + 5 min (9:35 AM)
    without any SIP delay cap, because the 9:30 AM bar is fixed historical
    data accessible at any wall-clock time.
    """

    def test_end_time_is_market_open_plus_5min(self):
        """
        end_time passed to get_bars must be 9:35 AM ET (market open + 5 min),
        not capped by now - SIP_DELAY_MINUTES.
        """
        system = _make_system()

        # Simulate 9:45 AM ET — with the OLD cap: min(9:35, 9:15) = 9:15 AM (wrong)
        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 9, 45, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            system._get_market_open_price('AAPL')

        ca = system.historical_client.get_bars.call_args
        end_dt = _parse_utc(ca.kwargs['end']).astimezone(ET_TZ)

        assert end_dt.hour == 9 and end_dt.minute == 35, (
            f"end_time was {end_dt.strftime('%H:%M')} ET, expected 09:35 ET. "
            f"The SIP delay cap should have been removed."
        )

    def test_start_time_is_market_open_minus_5min(self):
        """start_time passed to get_bars must be 9:25 AM ET."""
        system = _make_system()

        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 10, 30, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            system._get_market_open_price('AAPL')

        ca = system.historical_client.get_bars.call_args
        start_dt = _parse_utc(ca.kwargs['start']).astimezone(ET_TZ)

        assert start_dt.hour == 9 and start_dt.minute == 25, (
            f"start_time was {start_dt.strftime('%H:%M')} ET, expected 09:25 ET."
        )

    def test_end_time_not_capped_before_old_threshold(self):
        """
        Regression: with the old code, calling before 10:05 AM would cap
        end_time below start_time (9:25 AM). The new code must keep
        end_time = 9:35 AM regardless of current time.
        """
        system = _make_system()

        # 9:50 AM: old cap = min(9:35, 9:20) = 9:20 AM < start_time (9:25 AM)
        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 9, 50, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            system._get_market_open_price('AAPL')

        ca = system.historical_client.get_bars.call_args
        start_dt = _parse_utc(ca.kwargs['start'])
        end_dt = _parse_utc(ca.kwargs['end'])

        assert end_dt >= start_dt, (
            f"end_time ({end_dt}) is before start_time ({start_dt}). "
            f"The SIP delay cap was not properly removed."
        )

    def test_returns_open_price_when_bar_near_market_open(self):
        """Returns the open price of the bar closest to 9:30 AM ET."""
        system = _make_system()

        # Use a weekday (Tuesday 2026-03-10) so the code doesn't roll back to Friday
        bar_time = ET_TZ.localize(real_datetime(2026, 3, 10, 9, 30, 0))
        system.historical_client.get_bars.return_value = [
            _make_bar(bar_time, 42.50)
        ]

        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 10, 11, 0, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            result = system._get_market_open_price('AAPL')

        assert result == 42.50

    def test_returns_none_when_no_bars(self):
        """Returns None when the API returns no bars."""
        system = _make_system()
        system.historical_client.get_bars.return_value = []

        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 11, 0, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            result = system._get_market_open_price('AAPL')

        assert result is None

    def test_returns_none_when_bar_too_far_from_open(self):
        """Returns None when the closest bar is more than 5 minutes from 9:30 AM."""
        system = _make_system()

        # Bar at 9:20 AM — 10 minutes before open, outside the 5-min tolerance
        bar_time = ET_TZ.localize(real_datetime(2026, 3, 7, 9, 20, 0))
        system.historical_client.get_bars.return_value = [
            _make_bar(bar_time, 42.50)
        ]

        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 11, 0, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            result = system._get_market_open_price('AAPL')

        assert result is None

    def test_returns_none_when_no_client(self):
        """Returns None when historical_client is None."""
        system = _make_system()
        system.historical_client = None

        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 11, 0, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            result = system._get_market_open_price('AAPL')

        assert result is None

    def test_sip_feed_is_used(self):
        """get_bars must be called with feed='sip'."""
        system = _make_system()

        FakeDatetime._fake_now = ET_TZ.localize(real_datetime(2026, 3, 7, 11, 0, 0))

        with patch('code.momentum_alerts.datetime', FakeDatetime):
            system._get_market_open_price('AAPL')

        ca = system.historical_client.get_bars.call_args
        assert ca.kwargs.get('feed') == 'sip'
