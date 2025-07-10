"""
Notification sender atoms for alert delivery.

This atom provides multiple notification channels including email, 
Slack, webhook, and console notifications for alert delivery.
"""

import json
import smtplib
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from enum import Enum
import requests
from pathlib import Path


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    enabled: bool = True
    
    # Email configuration
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)
    
    # Slack configuration
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    slack_username: Optional[str] = "AlertBot"
    
    # Webhook configuration
    webhook_url: Optional[str] = None
    webhook_method: str = "POST"
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # File configuration
    file_path: Optional[str] = None
    
    # Rate limiting
    max_notifications_per_hour: int = 100
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 5


@dataclass
class NotificationMessage:
    """Notification message structure."""
    title: str
    content: str
    severity: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'content': self.content,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class NotificationSender:
    """
    Multi-channel notification sender for alerts.
    
    Supports email, Slack, webhook, console, and file notifications
    with retry logic and rate limiting.
    """
    
    def __init__(self, default_config: Optional[NotificationConfig] = None):
        """
        Initialize the notification sender.
        
        Args:
            default_config: Default notification configuration
        """
        self.configs: Dict[str, NotificationConfig] = {}
        self.notification_history: List[Dict[str, Any]] = []
        self.rate_limit_counters: Dict[str, List[datetime]] = {}
        
        if default_config:
            self.add_config("default", default_config)
        
        self.logger = logging.getLogger(__name__)
    
    def add_config(self, name: str, config: NotificationConfig) -> bool:
        """
        Add a notification configuration.
        
        Args:
            name: Name identifier for the configuration
            config: NotificationConfig object
            
        Returns:
            True if configuration was added successfully
        """
        self.configs[name] = config
        self.rate_limit_counters[name] = []
        self.logger.info(f"Added notification config: {name} ({config.channel.value})")
        return True
    
    def remove_config(self, name: str) -> bool:
        """
        Remove a notification configuration.
        
        Args:
            name: Name of the configuration to remove
            
        Returns:
            True if configuration was removed successfully
        """
        if name in self.configs:
            del self.configs[name]
            if name in self.rate_limit_counters:
                del self.rate_limit_counters[name]
            self.logger.info(f"Removed notification config: {name}")
            return True
        return False
    
    def send_notification(self, 
                         message: NotificationMessage,
                         config_name: str = "default") -> bool:
        """
        Send a notification using the specified configuration.
        
        Args:
            message: NotificationMessage to send
            config_name: Name of the configuration to use
            
        Returns:
            True if notification was sent successfully
        """
        if config_name not in self.configs:
            self.logger.error(f"Configuration '{config_name}' not found")
            return False
        
        config = self.configs[config_name]
        
        if not config.enabled:
            self.logger.debug(f"Configuration '{config_name}' is disabled")
            return True
        
        # Check rate limiting
        if not self._check_rate_limit(config_name, config):
            self.logger.warning(f"Rate limit exceeded for config '{config_name}'")
            return False
        
        # Send notification based on channel type
        success = False
        
        if config.channel == NotificationChannel.EMAIL:
            success = self._send_email(message, config)
        elif config.channel == NotificationChannel.SLACK:
            success = self._send_slack(message, config)
        elif config.channel == NotificationChannel.WEBHOOK:
            success = self._send_webhook(message, config)
        elif config.channel == NotificationChannel.CONSOLE:
            success = self._send_console(message, config)
        elif config.channel == NotificationChannel.FILE:
            success = self._send_file(message, config)
        
        # Record notification attempt
        self.notification_history.append({
            'timestamp': datetime.now().isoformat(),
            'config_name': config_name,
            'channel': config.channel.value,
            'success': success,
            'message_title': message.title,
            'message_severity': message.severity
        })
        
        # Update rate limit counter
        if success:
            self.rate_limit_counters[config_name].append(datetime.now())
        
        return success
    
    def send_to_all(self, message: NotificationMessage) -> Dict[str, bool]:
        """
        Send notification to all configured channels.
        
        Args:
            message: NotificationMessage to send
            
        Returns:
            Dictionary with config names and success status
        """
        results = {}
        
        for config_name in self.configs:
            results[config_name] = self.send_notification(message, config_name)
        
        return results
    
    def send_to_channels(self, 
                        message: NotificationMessage,
                        channels: List[NotificationChannel]) -> Dict[str, bool]:
        """
        Send notification to specific channel types.
        
        Args:
            message: NotificationMessage to send
            channels: List of channels to send to
            
        Returns:
            Dictionary with config names and success status
        """
        results = {}
        
        for config_name, config in self.configs.items():
            if config.channel in channels:
                results[config_name] = self.send_notification(message, config_name)
        
        return results
    
    def _check_rate_limit(self, config_name: str, config: NotificationConfig) -> bool:
        """Check if rate limit allows sending notification."""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self.rate_limit_counters[config_name] = [
            ts for ts in self.rate_limit_counters[config_name]
            if ts > one_hour_ago
        ]
        
        # Check if under limit
        return len(self.rate_limit_counters[config_name]) < config.max_notifications_per_hour
    
    def _send_email(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """Send email notification."""
        try:
            if not all([config.smtp_server, config.smtp_username, config.smtp_password,
                       config.email_from, config.email_to]):
                self.logger.error("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config.email_from
            msg['To'] = ', '.join(config.email_to)
            msg['Subject'] = f"[{message.severity.upper()}] {message.title}"
            
            # Email body
            body = f"""
Alert Notification

Title: {message.title}
Severity: {message.severity}
Timestamp: {message.timestamp}

Content:
{message.content}

Metadata:
{json.dumps(message.metadata, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config.smtp_server, config.smtp_port)
            server.starttls()
            server.login(config.smtp_username, config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email sent successfully to {config.email_to}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _send_slack(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """Send Slack notification."""
        try:
            if not config.slack_webhook_url:
                self.logger.error("Slack webhook URL not configured")
                return False
            
            # Create Slack message
            slack_message = {
                "channel": config.slack_channel,
                "username": config.slack_username,
                "text": f"*[{message.severity.upper()}] {message.title}*",
                "attachments": [
                    {
                        "color": self._get_color_for_severity(message.severity),
                        "fields": [
                            {
                                "title": "Content",
                                "value": message.content,
                                "short": False
                            },
                            {
                                "title": "Timestamp",
                                "value": message.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(
                config.slack_webhook_url,
                json=slack_message,
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info("Slack notification sent successfully")
                return True
            else:
                self.logger.error(f"Slack notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _send_webhook(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """Send webhook notification."""
        try:
            if not config.webhook_url:
                self.logger.error("Webhook URL not configured")
                return False
            
            # Prepare webhook payload
            payload = message.to_dict()
            
            # Send webhook
            if config.webhook_method.upper() == "POST":
                response = requests.post(
                    config.webhook_url,
                    json=payload,
                    headers=config.webhook_headers,
                    timeout=30
                )
            else:
                response = requests.get(
                    config.webhook_url,
                    params=payload,
                    headers=config.webhook_headers,
                    timeout=30
                )
            
            if response.status_code in [200, 201, 202]:
                self.logger.info("Webhook notification sent successfully")
                return True
            else:
                self.logger.error(f"Webhook notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _send_console(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """Send console notification."""
        try:
            severity_symbol = {
                'critical': '🔥',
                'warning': '⚠️',
                'info': 'ℹ️',
                'debug': '🔍'
            }.get(message.severity.lower(), '📢')
            
            console_message = f"""
{severity_symbol} [{message.severity.upper()}] {message.title}
Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Content: {message.content}
"""
            
            print(console_message)
            self.logger.info("Console notification sent")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send console notification: {e}")
            return False
    
    def _send_file(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """Send file notification."""
        try:
            if not config.file_path:
                self.logger.error("File path not configured")
                return False
            
            # Ensure directory exists
            Path(config.file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare file content
            file_content = {
                'timestamp': message.timestamp.isoformat(),
                'title': message.title,
                'severity': message.severity,
                'content': message.content,
                'metadata': message.metadata
            }
            
            # Append to file
            with open(config.file_path, 'a') as f:
                f.write(json.dumps(file_content) + '\n')
            
            self.logger.info(f"File notification written to {config.file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send file notification: {e}")
            return False
    
    def _get_color_for_severity(self, severity: str) -> str:
        """Get color code for Slack messages based on severity."""
        colors = {
            'critical': 'danger',
            'warning': 'warning',
            'info': 'good',
            'debug': '#36a64f'
        }
        return colors.get(severity.lower(), 'good')
    
    def get_notification_stats(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get notification statistics.
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Dictionary with notification statistics
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_notifications = [
            n for n in self.notification_history
            if datetime.fromisoformat(n['timestamp']) >= cutoff_time
        ]
        
        stats = {
            'total_notifications': len(recent_notifications),
            'successful_notifications': len([n for n in recent_notifications if n['success']]),
            'failed_notifications': len([n for n in recent_notifications if not n['success']]),
            'notifications_by_channel': {},
            'notifications_by_severity': {},
            'success_rate': 0.0
        }
        
        # Count by channel
        for notification in recent_notifications:
            channel = notification['channel']
            if channel not in stats['notifications_by_channel']:
                stats['notifications_by_channel'][channel] = 0
            stats['notifications_by_channel'][channel] += 1
        
        # Count by severity
        for notification in recent_notifications:
            severity = notification['message_severity']
            if severity not in stats['notifications_by_severity']:
                stats['notifications_by_severity'][severity] = 0
            stats['notifications_by_severity'][severity] += 1
        
        # Calculate success rate
        if stats['total_notifications'] > 0:
            stats['success_rate'] = stats['successful_notifications'] / stats['total_notifications']
        
        return stats
    
    def test_configuration(self, config_name: str) -> bool:
        """
        Test a notification configuration.
        
        Args:
            config_name: Name of the configuration to test
            
        Returns:
            True if test message was sent successfully
        """
        if config_name not in self.configs:
            return False
        
        test_message = NotificationMessage(
            title="Test Notification",
            content="This is a test notification to verify configuration.",
            severity="info",
            timestamp=datetime.now(),
            metadata={"test": True}
        )
        
        return self.send_notification(test_message, config_name)


def create_default_email_config(
    smtp_server: str,
    smtp_username: str,
    smtp_password: str,
    email_from: str,
    email_to: List[str]
) -> NotificationConfig:
    """
    Create default email notification configuration.
    
    Args:
        smtp_server: SMTP server address
        smtp_username: SMTP username
        smtp_password: SMTP password
        email_from: From email address
        email_to: List of recipient email addresses
        
    Returns:
        NotificationConfig for email
    """
    return NotificationConfig(
        channel=NotificationChannel.EMAIL,
        smtp_server=smtp_server,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        email_from=email_from,
        email_to=email_to
    )


def create_slack_config(webhook_url: str, channel: str = None) -> NotificationConfig:
    """
    Create Slack notification configuration.
    
    Args:
        webhook_url: Slack webhook URL
        channel: Slack channel (optional)
        
    Returns:
        NotificationConfig for Slack
    """
    return NotificationConfig(
        channel=NotificationChannel.SLACK,
        slack_webhook_url=webhook_url,
        slack_channel=channel
    )


def create_webhook_config(webhook_url: str, 
                         method: str = "POST",
                         headers: Dict[str, str] = None) -> NotificationConfig:
    """
    Create webhook notification configuration.
    
    Args:
        webhook_url: Webhook URL
        method: HTTP method (default: POST)
        headers: HTTP headers
        
    Returns:
        NotificationConfig for webhook
    """
    return NotificationConfig(
        channel=NotificationChannel.WEBHOOK,
        webhook_url=webhook_url,
        webhook_method=method,
        webhook_headers=headers or {}
    )


def create_console_config() -> NotificationConfig:
    """
    Create console notification configuration.
    
    Returns:
        NotificationConfig for console output
    """
    return NotificationConfig(
        channel=NotificationChannel.CONSOLE
    )


def create_file_config(file_path: str) -> NotificationConfig:
    """
    Create file notification configuration.
    
    Args:
        file_path: Path to notification log file
        
    Returns:
        NotificationConfig for file output
    """
    return NotificationConfig(
        channel=NotificationChannel.FILE,
        file_path=file_path
    )