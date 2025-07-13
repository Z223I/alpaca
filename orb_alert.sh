#!/bin/bash

# ORB Alert System Launcher
# 
# This script launches both the ORB alerts system and the alerts monitor
# with the same arguments. It handles process management and graceful shutdown.
#
# The script will:
# 1. Activate the 'alpaca' conda environment
# 2. Validate that all required files and dependencies exist
# 3. Launch code/orb_alerts.py (generates ORB breakout alerts)
# 4. Launch code/orb_alerts_monitor.py (monitors alerts and creates super alerts)
# 5. Monitor both processes and handle graceful shutdown on Ctrl+C
#
# Usage:
#   ./orb_alert.sh                                    # Use current date CSV file
#   ./orb_alert.sh --symbols-file data/20250625.csv   # Use specific symbols file
#   ./orb_alert.sh --test                            # Run in test mode
#   ./orb_alert.sh --verbose                         # Enable verbose logging
#   ./orb_alert.sh --summary                         # Show summary and exit
#
# Both systems will run simultaneously:
# - ORB Alerts: Monitors market data and generates bullish/bearish breakout alerts
# - Alerts Monitor: Watches for bullish alerts and creates super alerts when Signal price is reached

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Process IDs for cleanup
ALERTS_PID=""
MONITOR_PID=""

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print header
print_header() {
    echo
    print_colored $CYAN "=================================================================="
    print_colored $CYAN "                    ORB ALERT SYSTEM LAUNCHER"
    print_colored $CYAN "=================================================================="
    echo
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Launch both ORB alerts system and alerts monitor with synchronized arguments."
    echo
    echo "Options:"
    echo "  --symbols-file FILE    Path to symbols CSV file (default: data/YYYYMMDD.csv)"
    echo "  --test                 Run in test mode (dry run)"
    echo "  --verbose              Enable verbose logging"
    echo "  --summary              Show daily summary and exit"
    echo "  --help, -h             Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                        # Use current date file"
    echo "  $0 --symbols-file data/20250625.csv      # Use specific symbols file"
    echo "  $0 --test --verbose                      # Test mode with verbose output"
    echo "  $0 --summary                             # Show summary only"
    echo
}

# Function to activate conda environment
activate_conda() {
    print_colored $YELLOW "Activating conda environment 'alpaca'..."
    
    # Source conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        print_colored $RED "❌ Error: Could not find conda installation"
        exit 1
    fi
    
    # Activate alpaca environment
    if ! conda activate alpaca 2>/dev/null; then
        print_colored $RED "❌ Error: Could not activate 'alpaca' conda environment"
        print_colored $YELLOW "Please ensure the 'alpaca' environment exists and contains required dependencies"
        exit 1
    fi
    
    print_colored $GREEN "✅ Conda environment activated"
}

# Function to validate files exist
validate_environment() {
    print_colored $YELLOW "Validating environment..."
    
    # Check if we're in the correct directory
    if [ ! -f "code/orb_alerts.py" ] || [ ! -f "code/orb_alerts_monitor.py" ]; then
        print_colored $RED "❌ Error: ORB alert scripts not found"
        print_colored $YELLOW "Please run this script from the alpaca project root directory"
        exit 1
    fi
    
    # Check Python dependencies
    if ! python3 -c "import alpaca_trade_api, pandas, pytz, watchdog" 2>/dev/null; then
        print_colored $RED "❌ Error: Required Python dependencies not installed"
        print_colored $YELLOW "Please ensure all dependencies are installed in the 'alpaca' conda environment"
        exit 1
    fi
    
    print_colored $GREEN "✅ Environment validation passed"
}

# Function to handle cleanup on exit
cleanup() {
    print_colored $YELLOW "\n🛑 Shutdown signal received, cleaning up..."
    
    # Kill the alerts system
    if [ ! -z "$ALERTS_PID" ] && kill -0 "$ALERTS_PID" 2>/dev/null; then
        print_colored $YELLOW "Stopping ORB alerts system (PID: $ALERTS_PID)..."
        kill -TERM "$ALERTS_PID" 2>/dev/null || true
        wait "$ALERTS_PID" 2>/dev/null || true
    fi
    
    # Kill the monitor
    if [ ! -z "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        print_colored $YELLOW "Stopping ORB alerts monitor (PID: $MONITOR_PID)..."
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    
    print_colored $GREEN "✅ Cleanup completed"
    exit 0
}

# Function to start ORB alerts system
start_alerts_system() {
    local args="$1"
    
    print_colored $BLUE "🚀 Starting ORB Alerts System..."
    print_colored $CYAN "Command: python3 code/orb_alerts.py $args"
    echo
    
    # Start the alerts system in background
    python3 code/orb_alerts.py $args &
    ALERTS_PID=$!
    
    print_colored $GREEN "✅ ORB Alerts System started (PID: $ALERTS_PID)"
}

# Function to start alerts monitor
start_alerts_monitor() {
    local args="$1"
    
    print_colored $BLUE "🔍 Starting ORB Alerts Monitor..."
    print_colored $CYAN "Command: python3 code/orb_alerts_monitor.py $args"
    echo
    
    # Start the monitor in background
    python3 code/orb_alerts_monitor.py $args &
    MONITOR_PID=$!
    
    print_colored $GREEN "✅ ORB Alerts Monitor started (PID: $MONITOR_PID)"
}

# Function to monitor processes
monitor_processes() {
    print_colored $PURPLE "\n📊 Both systems are running. Press Ctrl+C to stop."
    print_colored $PURPLE "Monitoring process health..."
    echo
    
    while true; do
        # Check if alerts system is still running
        if [ ! -z "$ALERTS_PID" ] && ! kill -0 "$ALERTS_PID" 2>/dev/null; then
            print_colored $RED "❌ ORB Alerts System stopped unexpectedly"
            cleanup
            exit 1
        fi
        
        # Check if monitor is still running
        if [ ! -z "$MONITOR_PID" ] && ! kill -0 "$MONITOR_PID" 2>/dev/null; then
            print_colored $RED "❌ ORB Alerts Monitor stopped unexpectedly"
            cleanup
            exit 1
        fi
        
        sleep 5
    done
}

# Function to show summary only
show_summary() {
    local args="$1"
    
    print_colored $BLUE "📈 Showing ORB Alerts Summary..."
    python3 code/orb_alerts.py --summary $args
    exit 0
}

# Main function
main() {
    # Parse arguments
    ARGS=""
    SUMMARY_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                print_usage
                exit 0
                ;;
            --summary)
                SUMMARY_MODE=true
                ARGS="$ARGS $1"
                shift
                ;;
            --symbols-file)
                if [ -z "$2" ]; then
                    print_colored $RED "❌ Error: --symbols-file requires a file path"
                    exit 1
                fi
                ARGS="$ARGS $1 $2"
                shift 2
                ;;
            --test|--verbose)
                ARGS="$ARGS $1"
                shift
                ;;
            *)
                print_colored $RED "❌ Error: Unknown option '$1'"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Print header
    print_header
    
    # Activate conda environment
    activate_conda
    
    # Validate environment
    validate_environment
    
    # Handle summary mode
    if [ "$SUMMARY_MODE" = true ]; then
        show_summary "$ARGS"
    fi
    
    # Set up signal handlers for graceful shutdown
    trap cleanup SIGINT SIGTERM
    
    # Start both systems
    print_colored $PURPLE "🎯 Launching ORB Alert System with arguments: $ARGS"
    echo
    
    start_alerts_system "$ARGS"
    sleep 2  # Give alerts system time to start
    
    start_alerts_monitor "$ARGS"
    sleep 2  # Give monitor time to start
    
    # Monitor the processes
    monitor_processes
}

# Run main function with all arguments
main "$@"