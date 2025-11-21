#!/bin/bash
#
# Setup script for Squeeze Alerts systemd service
#
# This script installs and enables the squeeze_alerts.service as a user service
# so it starts automatically on system boot and restarts on failure.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$SCRIPT_DIR/squeeze_alerts.service"
USER_SERVICE_DIR="$HOME/.config/systemd/user"

echo "=========================================="
echo "Squeeze Alerts Service Setup"
echo "=========================================="
echo ""

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Error: Service file not found: $SERVICE_FILE"
    exit 1
fi

# Create logs directory if it doesn't exist
LOGS_DIR="$PROJECT_ROOT/logs"
if [ ! -d "$LOGS_DIR" ]; then
    echo "📁 Creating logs directory: $LOGS_DIR"
    mkdir -p "$LOGS_DIR"
fi

# Create user systemd directory if it doesn't exist
if [ ! -d "$USER_SERVICE_DIR" ]; then
    echo "📁 Creating systemd user directory: $USER_SERVICE_DIR"
    mkdir -p "$USER_SERVICE_DIR"
fi

# Copy service file to systemd user directory
echo "📋 Installing service file..."
cp "$SERVICE_FILE" "$USER_SERVICE_DIR/"
echo "   Copied to: $USER_SERVICE_DIR/squeeze_alerts.service"

# Reload systemd user daemon
echo "🔄 Reloading systemd user daemon..."
systemctl --user daemon-reload

# Enable the service (start on boot)
echo "✅ Enabling service (auto-start on boot)..."
systemctl --user enable squeeze_alerts.service

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Service Commands:"
echo "  Start:   systemctl --user start squeeze_alerts"
echo "  Stop:    systemctl --user stop squeeze_alerts"
echo "  Status:  systemctl --user status squeeze_alerts"
echo "  Logs:    journalctl --user -u squeeze_alerts -f"
echo ""
echo "Log Files:"
echo "  Output:  $LOGS_DIR/squeeze_alerts.log"
echo "  Errors:  $LOGS_DIR/squeeze_alerts_error.log"
echo ""
echo "To start the service now, run:"
echo "  systemctl --user start squeeze_alerts"
echo ""
