#!/bin/bash

# Purpose: Run 80-class object detection feature extraction in background with monitoring
# Arguments: None (uses default configuration)
# Returns: Background process with monitoring capabilities

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="extract_detection_features_80_classes.py"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/80class_extraction.pid"
LOG_FILE="$LOG_DIR/80class_extraction.log"
PROGRESS_FILE="$SCRIPT_DIR/../output/progress_80_classes.json"
TEMP_FEATURES_FILE="$SCRIPT_DIR/../output/temp_detection_features_80_classes.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # Process not running, clean up stale PID file
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to start the extraction
start_extraction() {
    if is_running; then
        print_warning "80-class detection extraction is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    
    print_status "Starting 80-class object detection extraction in background..."
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Activate conda environment and start the Python script in background with test_mode=False
    nohup bash -c "
source ~/.bashrc
conda activate dinov3_h200
cd '$SCRIPT_DIR'
python3 -c \"
import sys
sys.path.append('$SCRIPT_DIR')
from extract_detection_features_80_classes import run_feature_extraction
run_feature_extraction(test_mode=False, resume=True)
\"
" > "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # Save PID to file
    echo "$pid" > "$PID_FILE"
    
    print_success "80-class detection extraction started with PID: $pid"
    print_status "Log file: $LOG_FILE"
    print_status "PID file: $PID_FILE"
    print_status "Use './run_80class_extraction.sh status' to monitor progress"
    
    return 0
}

# Function to start test extraction
start_test_extraction() {
    if is_running; then
        print_warning "80-class detection extraction is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    
    print_status "Starting TEST 80-class detection extraction in background..."
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Activate conda environment and start the Python script in background with test_mode=True
    nohup bash -c "
source ~/.bashrc
conda activate dinov3_h200
cd '$SCRIPT_DIR'
python3 -c \"
import sys
sys.path.append('$SCRIPT_DIR')
from extract_detection_features_80_classes import run_feature_extraction
run_feature_extraction(test_mode=True, resume=False)
\"
" > "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # Save PID to file
    echo "$pid" > "$PID_FILE"
    
    print_success "TEST 80-class detection extraction started with PID: $pid"
    print_status "Log file: $LOG_FILE"
    print_status "PID file: $PID_FILE"
    print_status "Use './run_80class_extraction.sh status' to monitor progress"
    
    return 0
}

# Function to stop the extraction
stop_extraction() {
    if ! is_running; then
        print_warning "No 80-class detection extraction process found"
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    print_status "Stopping 80-class detection extraction (PID: $pid)..."
    
    # Send SIGTERM first
    kill -TERM "$pid" 2>/dev/null
    
    # Wait a bit for graceful shutdown
    sleep 5
    
    # Check if still running and force kill if necessary
    if ps -p "$pid" > /dev/null 2>&1; then
        print_warning "Process still running, force killing..."
        kill -KILL "$pid" 2>/dev/null
        sleep 2
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    print_success "80-class detection extraction stopped"
    return 0
}

# Function to show status and progress
show_status() {
    if ! is_running; then
        print_warning "No 80-class detection extraction process found"
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    print_status "80-class detection extraction is running (PID: $pid)"
    
    # Show process info
    echo
    print_status "Process Information:"
    ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem --no-headers
    
    # Show log tail
    echo
    print_status "Recent Log Output (last 20 lines):"
    if [ -f "$LOG_FILE" ]; then
        tail -n 20 "$LOG_FILE"
    else
        print_warning "Log file not found"
    fi
    
    # Show progress if available
    echo
    if [ -f "$PROGRESS_FILE" ]; then
        print_status "Progress Information:"
        if command -v jq >/dev/null 2>&1; then
            # Use jq if available for pretty formatting
            jq '.' "$PROGRESS_FILE"
        else
            # Fallback to cat
            cat "$PROGRESS_FILE"
        fi
    else
        print_warning "Progress file not found"
    fi
    
    # Show temporary features file info
    if [ -f "$TEMP_FEATURES_FILE" ]; then
        echo
        print_status "Temporary Features File:"
        local line_count=$(wc -l < "$TEMP_FEATURES_FILE")
        local file_size=$(du -h "$TEMP_FEATURES_FILE" | cut -f1)
        echo "   Lines: $line_count (including header)"
        echo "   Size: $file_size"
        echo "   Last modified: $(stat -c %y "$TEMP_FEATURES_FILE")"
    fi
    
    # Show output CSV files
    echo
    print_status "Output Files:"
    for file in "$SCRIPT_DIR/../output"/*detection_features_80_classes.csv; do
        if [ -f "$file" ]; then
            local line_count=$(wc -l < "$file")
            local file_size=$(du -h "$file" | cut -f1)
            local filename=$(basename "$file")
            echo "   $filename: $line_count lines, $file_size"
        fi
    done
    
    return 0
}

# Function to show real-time monitoring
monitor() {
    if ! is_running; then
        print_warning "No 80-class detection extraction process found"
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    print_status "Starting real-time monitoring (PID: $pid)"
    print_status "Press Ctrl+C to stop monitoring"
    echo
    
    # Monitor in real-time
    tail -f "$LOG_FILE" &
    local tail_pid=$!
    
    # Trap Ctrl+C to clean up
    trap "kill $tail_pid 2>/dev/null; exit 0" INT
    
    # Wait for tail process
    wait $tail_pid
}

# Function to show help
show_help() {
    echo "80-Class Object Detection Feature Extraction Runner"
    echo "=================================================="
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     Start FULL 80-class detection extraction in background (all images)"
    echo "  test      Start TEST 80-class detection extraction in background (few images)"
    echo "  stop      Stop running 80-class detection extraction"
    echo "  restart   Restart 80-class detection extraction"
    echo "  status    Show current status and progress"
    echo "  monitor   Show real-time log monitoring"
    echo "  logs      Show recent log output"
    echo "  help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start          # Start full extraction (all images)"
    echo "  $0 test           # Start test extraction (few images)"
    echo "  $0 status         # Check progress"
    echo "  $0 monitor        # Real-time monitoring"
    echo "  $0 stop           # Stop extraction"
    echo
    echo "Notes:"
    echo "  - 'start' runs on ALL images with resume capability"
    echo "  - 'test' runs on few images for testing (no resume)"
    echo "  - Both run in background with auto-save every 100 images"
    echo "  - Extracts all 80 COCO object detection classes"
    echo
    echo "Files:"
    echo "  PID: $PID_FILE"
    echo "  Log: $LOG_FILE"
    echo "  Progress: $PROGRESS_FILE"
    echo "  Temp Features: $TEMP_FEATURES_FILE"
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        print_status "Recent Log Output (last 50 lines):"
        tail -n 50 "$LOG_FILE"
    else
        print_warning "Log file not found"
    fi
}

# Function to restart extraction
restart_extraction() {
    print_status "Restarting 80-class detection extraction..."
    stop_extraction
    sleep 2
    
    # Check if we should restart in test mode or full mode
    if [ -f "$PROGRESS_FILE" ]; then
        # Try to determine mode from progress file
        if grep -q '"test_mode": true' "$PROGRESS_FILE" 2>/dev/null; then
            print_status "Restarting in TEST mode..."
            start_test_extraction
        else:
            print_status "Restarting in FULL mode..."
            start_extraction
        fi
    else
        # Default to full mode if no progress file
        print_status "Restarting in FULL mode (default)..."
        start_extraction
    fi
}

# Main script logic
case "${1:-help}" in
    start)
        start_extraction
        ;;
    test)
        start_test_extraction
        ;;
    stop)
        stop_extraction
        ;;
    restart)
        restart_extraction
        ;;
    status)
        show_status
        ;;
    monitor)
        monitor
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
