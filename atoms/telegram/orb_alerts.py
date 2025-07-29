"""
Telegram ORB Alerts Atom

This atom reads super alert JSON files and sends the alert_message field via Telegram.
Designed to work with ORB super alert files from the historical_data directory.
"""

import json
import os
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