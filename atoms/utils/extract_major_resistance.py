"""
Utility to extract major resistance from volume profile JSON.
"""

import json
import os
from datetime import date
from typing import List, Optional


def extract_major_resistance(
        symbol: str,
        target_date: Optional[date] = None) -> List[float]:
    """
    Extract major resistance levels from volume profile JSON output.

    Major resistance is defined as the list of profile_high values from
    all profiles in the volume profile analysis.

    Args:
        symbol: Stock symbol
        target_date: Date for the volume profile data (default: today)

    Returns:
        List of major resistance price levels (empty list if not found)
    """
    if target_date is None:
        target_date = date.today()

    # Determine date string for file path (YYYYMMDD format)
    date_str = target_date.strftime('%Y%m%d')

    # Calculate previous day for volume profile filename
    # Volume profile is typically generated with data from previous day
    from datetime import timedelta
    prev_date = target_date - timedelta(days=1)
    prev_date_str = prev_date.strftime('%Y%m%d')

    # Try multiple possible paths for volume profile JSON
    # Format: ./historical_data/YYYY-MM-DD/volume_profile_output/
    #         SYMBOL_volume_profile_YYYYMMDD.json
    date_fmt = target_date.strftime('%Y-%m-%d')
    prev_date_fmt = prev_date.strftime('%Y-%m-%d')
    vp_dir = "volume_profile_output"

    possible_paths = [
        # Current date paths (try current date file first for intraday)
        f"./historical_data/{date_fmt}/{vp_dir}/"
        f"{symbol}_volume_profile_{date_str}.json",
        f"./historical_data/{date_fmt}/{vp_dir}/"
        f"{symbol}_volume_profile_{prev_date_str}.json",
        # Previous date paths
        f"./historical_data/{prev_date_fmt}/{vp_dir}/"
        f"{symbol}_volume_profile_{prev_date_str}.json",
        # Root volume_profile_output directory
        f"./{vp_dir}/{symbol}_volume_profile_{date_str}.json",
        f"./{vp_dir}/{symbol}_volume_profile_{prev_date_str}.json",
    ]

    json_path = None
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break

    if json_path is None:
        msg = (f"⚠ No volume profile JSON found for {symbol} "
               f"on {target_date}")
        print(msg)
        print("  Searched paths:")
        for path in possible_paths:
            print(f"    - {path}")
        return []

    try:
        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract profile_high from each profile
        profiles = data.get('profiles', [])
        major_resistance = []

        for profile in profiles:
            profile_high = profile.get('profile_high')
            if profile_high is not None:
                major_resistance.append(float(profile_high))

        if major_resistance:
            # Return only the highest resistance level
            highest_resistance = max(major_resistance)
            print(f"✓ Extracted major resistance from {json_path}")
            print(f"  All resistance levels: "
                  f"{[f'${level:.2f}' for level in major_resistance]}")
            print(f"  Highest resistance (plotted): "
                  f"${highest_resistance:.2f}")
            return [highest_resistance]
        else:
            print(f"⚠ No profile_high values found in {json_path}")
            return []

    except Exception as e:
        print(f"✗ Error reading volume profile JSON for {symbol}: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    # Test with IRBT
    print("Testing extract_major_resistance with IRBT...")
    resistance_levels = extract_major_resistance('IRBT', date(2025, 10, 15))
    print(f"\nResult: {resistance_levels}")

    if resistance_levels:
        print("✅ Test passed!")
    else:
        print("❌ Test failed - no resistance levels found")
