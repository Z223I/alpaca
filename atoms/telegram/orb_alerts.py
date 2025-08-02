"""
Telegram ORB Alerts Atom

This atom reads super alert JSON files and sends the alert_message field via Telegram.
Designed to work with ORB super alert files from the historical_data directory.
"""

import json
import os
import shutil
from typing import Optional, Dict, Any
from datetime import datetime

from .telegram_post import send_message

def send_orb_alert(file_path: str, urgent: bool = True, post_only_urgent: bool = False) -> Dict[str, Any]:
    """
    Send ORB alert from super alert JSON file.

    Args:
        file_path (str): Path to super alert JSON file
        urgent (bool): Whether to mark the message as urgent (default True)
        post_only_urgent (bool): If True, only send urgent messages (default False)

    Returns:
        dict: Result from send_message with additional metadata
    """
    try:
        # Validate file path
        if not _validate_file_path(file_path):
            return {
                'success': False,
                'error': 'Invalid file path or file does not exist',
                'file_path': file_path
            }

        # Load and parse JSON
        alert_data = _load_alert_file(file_path)
        if not alert_data:
            return {
                'success': False,
                'error': 'Failed to load or parse JSON file',
                'file_path': file_path
            }

        # Extract alert message
        alert_message = _extract_alert_message(alert_data)
        if not alert_message:
            return {
                'success': False,
                'error': 'No alert_message found in JSON file',
                'file_path': file_path
            }

        # Add ORB alert emoji and formatting
        formatted_message = _format_orb_message(alert_message, alert_data)

        # Check if we should send based on urgency filter
        if post_only_urgent and not urgent:
            return {
                'success': True,
                'skipped': True,
                'reason': 'Non-urgent message filtered by post_only_urgent setting',
                'file_path': file_path,
                'symbol': alert_data.get('symbol', 'UNKNOWN'),
                'timestamp': alert_data.get('timestamp', 'UNKNOWN'),
                'alert_type': alert_data.get('alert_type', 'UNKNOWN'),
                'urgent': urgent,
                'sent_count': 0
            }

        # Send via Telegram
        result = send_message(formatted_message, urgent=urgent)

        # Add metadata to result
        result.update({
            'file_path': file_path,
            'symbol': alert_data.get('symbol', 'UNKNOWN'),
            'timestamp': alert_data.get('timestamp', 'UNKNOWN'),
            'alert_type': alert_data.get('alert_type', 'UNKNOWN'),
            'original_message': alert_message
        })

        # Store superduper alerts that were actually sent
        if result.get('success', False) and not result.get('skipped', False):
            _store_sent_superduper_alert(file_path, alert_data)

        return result

    except Exception as e:
        return {
            'success': False,
            'error': f'Exception occurred: {str(e)}',
            'file_path': file_path
        }

def _validate_file_path(file_path: str) -> bool:
    """Validate that the file exists and has correct extension."""
    if not file_path:
        return False

    if not os.path.exists(file_path):
        return False

    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ['.json']:
        return False

    return True

def _load_alert_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load and parse JSON alert file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
        print(f"Error loading alert file {file_path}: {e}")
        return None

def _extract_alert_message(alert_data: Dict[str, Any]) -> Optional[str]:
    """Extract alert_message from JSON data, supporting nested structure."""
    # First try direct access
    alert_message = alert_data.get('alert_message')
    if alert_message:
        return alert_message

    # Try nested in original_alert
    original_alert = alert_data.get('original_alert', {})
    if isinstance(original_alert, dict):
        alert_message = original_alert.get('alert_message')
        if alert_message:
            return alert_message

    # Try other common locations
    for key in ['message', 'text', 'alert_text', 'notification']:
        if key in alert_data and alert_data[key]:
            return alert_data[key]

    return None

def _format_orb_message(alert_message: str, alert_data: Dict[str, Any]) -> str:
    """Format the ORB alert message for Telegram."""
    # Replace escaped newlines with actual newlines
    alert_message = alert_message.replace('\\n', '\n')

    # Add ORB alert prefix
    formatted = f"🚀 **ORB ALERT**\n\n{alert_message}"

    # Add metadata if available
    symbol = alert_data.get('symbol', '')
    timestamp = alert_data.get('timestamp', '')
    alert_type = alert_data.get('alert_type', '')

    if symbol or timestamp or alert_type:
        formatted += "\n\n📊 **Alert Details:**"

        if symbol:
            formatted += f"\n• Symbol: {symbol}"

        if timestamp:
            # Format timestamp nicely
            try:
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%H:%M:%S ET')
                    formatted += f"\n• Time: {formatted_time}"
            except:
                formatted += f"\n• Time: {timestamp}"

        if alert_type and alert_type != 'super_alert':
            formatted += f"\n• Type: {alert_type.replace('_', ' ').title()}"

    # Add risk information if available
    risk_info = _extract_risk_info(alert_data)
    if risk_info:
        formatted += f"\n\n⚠️ **Risk Info:**\n{risk_info}"

    return formatted

def _extract_risk_info(alert_data: Dict[str, Any]) -> Optional[str]:
    """Extract risk/trading information from alert data."""
    risk_parts = []

    # From original alert
    original_alert = alert_data.get('original_alert', {})
    if isinstance(original_alert, dict):
        stop_loss = original_alert.get('recommended_stop_loss')
        take_profit = original_alert.get('recommended_take_profit')
        confidence = original_alert.get('confidence_score')

        if stop_loss:
            risk_parts.append(f"Stop Loss: ${stop_loss:.2f}")
        if take_profit:
            risk_parts.append(f"Take Profit: ${take_profit:.2f}")
        if confidence:
            risk_parts.append(f"Confidence: {confidence:.2f}")

    # From risk assessment
    risk_assessment = alert_data.get('risk_assessment', {})
    if isinstance(risk_assessment, dict):
        entry_price = risk_assessment.get('entry_price')
        target_price = risk_assessment.get('target_price')
        risk_reward = risk_assessment.get('current_risk_reward')

        if entry_price:
            risk_parts.append(f"Entry: ${entry_price:.2f}")
        if target_price:
            risk_parts.append(f"Target: ${target_price:.2f}")
        if risk_reward:
            risk_parts.append(f"R/R: {risk_reward:.1f}")

    return " | ".join(risk_parts) if risk_parts else None

def _store_sent_superduper_alert(file_path: str, alert_data: Dict[str, Any]) -> None:
    """
    Store superduper alerts that were actually sent to the historical data directory.
    
    Directory structure: historical_data/YYYY-MM-DD/superduper_alerts_sent/{bullish,bearish}/{yellow,green}
    
    Args:
        file_path (str): Original path to the alert file
        alert_data (dict): The alert data that was sent
    """
    try:
        # Extract target date from file path (historical_data/YYYY-MM-DD/...)
        target_date = _extract_date_from_path(file_path)
        
        # Extract alert properties to determine directory placement
        alert_type = alert_data.get('alert_type', '').lower()
        symbol = alert_data.get('symbol', 'UNKNOWN')
        
        # Determine sentiment (bullish/bearish) from alert data
        sentiment = _determine_alert_sentiment(alert_data)
        if not sentiment:
            return  # Skip if we can't determine sentiment
        
        # Determine alert level (yellow/green) from alert data  
        alert_level = _determine_alert_level(alert_data)
        if not alert_level:
            return  # Skip if we can't determine alert level
        
        # Create target directory structure
        base_dir = f"historical_data/{target_date}/superduper_alerts_sent"
        target_dir = os.path.join(base_dir, sentiment, alert_level)
        
        # Create directories if they don't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Generate target filename
        original_filename = os.path.basename(file_path)
        target_file = os.path.join(target_dir, original_filename)
        
        # Copy the alert file to the sent directory
        shutil.copy2(file_path, target_file)
        
        print(f"Stored sent superduper alert: {target_file}")
        
    except Exception as e:
        print(f"Error storing sent superduper alert: {e}")

def _determine_alert_sentiment(alert_data: Dict[str, Any]) -> Optional[str]:
    """
    Determine if alert is bullish or bearish from alert data.
    
    Returns:
        str: 'bullish' or 'bearish', or None if cannot determine
    """
    # Check alert message for sentiment indicators
    alert_message = alert_data.get('alert_message', '').lower()
    
    # Look for bullish indicators
    bullish_indicators = ['buy', 'long', 'bullish', 'upward', 'breakout', 'above', 'bull', 'green']
    bearish_indicators = ['sell', 'short', 'bearish', 'downward', 'breakdown', 'below', 'bear', 'red']
    
    bullish_score = sum(1 for indicator in bullish_indicators if indicator in alert_message)
    bearish_score = sum(1 for indicator in bearish_indicators if indicator in alert_message)
    
    if bullish_score > bearish_score:
        return 'bullish'
    elif bearish_score > bullish_score:
        return 'bearish'
    
    # Check in original_alert if available
    original_alert = alert_data.get('original_alert', {})
    if isinstance(original_alert, dict):
        original_message = original_alert.get('alert_message', '').lower()
        
        bullish_score = sum(1 for indicator in bullish_indicators if indicator in original_message)
        bearish_score = sum(1 for indicator in bearish_indicators if indicator in original_message)
        
        if bullish_score > bearish_score:
            return 'bullish'
        elif bearish_score > bullish_score:
            return 'bearish'
    
    # Default fallback - could be enhanced with more sophisticated logic
    return None

def _determine_alert_level(alert_data: Dict[str, Any]) -> Optional[str]:
    """
    Determine if alert is yellow or green level from the actual light indicators.
    Superduper alerts already have green/yellow lights based on trend analysis and momentum.
    
    Returns:
        str: 'yellow' or 'green', or None if cannot determine
    """
    # Check alert message for light indicators
    alert_message = alert_data.get('alert_message', '').lower()
    
    # Look for light indicators
    if 'green light' in alert_message or '🟢' in alert_message or 'green🔥' in alert_message:
        return 'green'
    elif 'yellow light' in alert_message or '🟡' in alert_message or 'yellow🔥' in alert_message:
        return 'yellow'
    
    # Check for color words
    if 'green' in alert_message and 'yellow' not in alert_message:
        return 'green'
    elif 'yellow' in alert_message and 'green' not in alert_message:
        return 'yellow'
    
    # Check in original_alert if available
    original_alert = alert_data.get('original_alert', {})
    if isinstance(original_alert, dict):
        original_message = original_alert.get('alert_message', '').lower()
        
        if 'green light' in original_message or '🟢' in original_message:
            return 'green'
        elif 'yellow light' in original_message or '🟡' in original_message:
            return 'yellow'
        
        if 'green' in original_message and 'yellow' not in original_message:
            return 'green'
        elif 'yellow' in original_message and 'green' not in original_message:
            return 'yellow'
    
    # Check for any trend/momentum indicators that might specify color
    trend_info = alert_data.get('trend_analysis', {})
    if isinstance(trend_info, dict):
        momentum_color = trend_info.get('momentum_color', '').lower()
        if momentum_color in ['green', 'yellow']:
            return momentum_color
    
    return None

def _extract_date_from_path(file_path: str) -> str:
    """
    Extract date from file path in format historical_data/YYYY-MM-DD/...
    
    Args:
        file_path (str): Path to the alert file
        
    Returns:
        str: Date in YYYY-MM-DD format, or current date if not found
    """
    import re
    
    # Look for date pattern in path
    date_match = re.search(r'historical_data/(\d{4}-\d{2}-\d{2})/', file_path)
    
    if date_match:
        return date_match.group(1)
    
    # Fallback to current date if no date found in path
    return datetime.now().strftime('%Y-%m-%d')