#!/bin/bash

# =============================================================================
# Enhanced Experiment Monitor with Progress Tracking
# =============================================================================
# Real-time monitoring of delta sensitivity experiments with detailed progress
# =============================================================================

# Configuration
if [ -z "${BASE_DIR}" ]; then
    export BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

EXPERIMENTS_DIR="${BASE_DIR}/experiments"
REFRESH_INTERVAL=10

# Model and delta configurations (must match experiment settings)
TOTAL_PERCEPTIONS=4  # beautiful, lively, boring, safe
TOTAL_DELTAS=7       # 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8
TOTAL_MODELS=4       # random_forest, svm, xgboost, mlp
TOTAL_EXPERIMENTS=$((TOTAL_PERCEPTIONS * TOTAL_DELTAS * TOTAL_MODELS))

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# FUNCTIONS
# =============================================================================

print_header() {
    clear
    echo "========================================================================"
    echo "          DELTA SENSITIVITY EXPERIMENT MONITOR"
    echo "========================================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Base: ${BASE_DIR}"
    echo ""
}

get_progress_bar() {
    local current=$1
    local total=$2
    local width=30
    
    if [ $total -eq 0 ]; then
        echo "[No data]"
        return
    fi
    
    local percent=$((current * 100 / total))
    local filled=$((percent * width / 100))
    
    printf "["
    for ((i=0; i<filled; i++)); do printf "="; done
    for ((i=filled; i<width; i++)); do printf " "; done
    printf "] %3d%%" $percent
}

parse_log_progress() {
    local log_file=$1
    
    if [ ! -f "$log_file" ]; then
        echo "0 0 0 - - -"
        return
    fi
    
    # Count completed items
    local completed_models=$(grep -c "Model.*trained\|trained in.*seconds" "$log_file" 2>/dev/null || echo 0)
    local completed_perceptions=$(grep -c "Completed perception\|✓.*perception" "$log_file" 2>/dev/null || echo 0)
    local completed_deltas=$(grep -c "Completed delta\|✓.*delta" "$log_file" 2>/dev/null || echo 0)
    
    # Get current processing items
    local current_perception=$(grep "Processing perception:" "$log_file" 2>/dev/null | tail -1 | sed 's/.*Processing perception: //' | cut -d' ' -f1)
    local current_delta=$(grep "Processing delta:" "$log_file" 2>/dev/null | tail -1 | sed 's/.*Processing delta: //')
    local current_model=$(grep "Training.*model\|Training [a-z_]*" "$log_file" 2>/dev/null | tail -1 | sed 's/.*Training //' | cut -d' ' -f1)
    
    echo "$completed_models $completed_perceptions $completed_deltas ${current_perception:-none} ${current_delta:-none} ${current_model:-none}"
}

format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

estimate_remaining_time() {
    local completed=$1
    local total=$2
    local elapsed=$3
    
    if [ $completed -eq 0 ] || [ $total -eq 0 ]; then
        echo "Calculating..."
        return
    fi
    
    local rate=$((elapsed / completed))
    local remaining=$((rate * (total - completed)))
    
    format_time $remaining
}

show_experiment_details() {
    local exp_dir=$1
    local pid=$2
    local log_file=$3
    
    # Get runtime
    local runtime=$(ps -o etime= -p $pid 2>/dev/null | tr -d ' ' || echo "00:00")
    local runtime_seconds=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ' || echo 0)
    
    # Get system resources
    local cpu=$(ps -o %cpu= -p $pid 2>/dev/null | tr -d ' ' || echo "0")
    local mem=$(ps -o %mem= -p $pid 2>/dev/null | tr -d ' ' || echo "0")
    
    # Parse log for progress
    read completed_models completed_perceptions completed_deltas current_perception current_delta current_model <<< $(parse_log_progress "$log_file")
    
    # Display experiment info
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC} Experiment: $(basename "$exp_dir")"
    echo -e "${GREEN}║${NC} PID: $pid | Runtime: $runtime | CPU: ${cpu}% | Memory: ${mem}%"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════════╣${NC}"
    
    # Overall progress
    echo -e "${GREEN}║${NC} ${BLUE}Overall Progress:${NC}"
    echo -e "${GREEN}║${NC}   Models: $completed_models/$TOTAL_EXPERIMENTS $(get_progress_bar $completed_models $TOTAL_EXPERIMENTS)"
    
    # Detailed progress
    echo -e "${GREEN}║${NC}"
    echo -e "${GREEN}║${NC} ${BLUE}Current Activity:${NC}"
    
    if [ "$current_perception" != "none" ] && [ "$current_perception" != "-" ]; then
        echo -e "${GREEN}║${NC}   Perception: ${YELLOW}$current_perception${NC}"
    fi
    
    if [ "$current_delta" != "none" ] && [ "$current_delta" != "-" ]; then
        echo -e "${GREEN}║${NC}   Delta: ${YELLOW}$current_delta${NC}"
    fi
    
    if [ "$current_model" != "none" ] && [ "$current_model" != "-" ]; then
        echo -e "${GREEN}║${NC}   Model: ${YELLOW}$current_model${NC}"
    fi
    
    # Time estimation
    if [ $runtime_seconds -gt 0 ] && [ $completed_models -gt 0 ]; then
        local eta=$(estimate_remaining_time $completed_models $TOTAL_EXPERIMENTS $runtime_seconds)
        echo -e "${GREEN}║${NC}"
        echo -e "${GREEN}║${NC} ${BLUE}Time Estimate:${NC}"
        echo -e "${GREEN}║${NC}   ETA: $eta"
    fi
    
    # Recent log activity
    echo -e "${GREEN}║${NC}"
    echo -e "${GREEN}║${NC} ${BLUE}Recent Activity:${NC}"
    if [ -f "$log_file" ]; then
        tail -n 3 "$log_file" 2>/dev/null | while IFS= read -r line; do
            # Truncate long lines
            if [ ${#line} -gt 65 ]; then
                line="${line:0:62}..."
            fi
            echo -e "${GREEN}║${NC}   $line"
        done
    fi
    
    # Check for errors
    local error_count=$(grep -c "ERROR\|FAILED\|Exception" "$log_file" 2>/dev/null || echo 0)
    if [ $error_count -gt 0 ]; then
        echo -e "${GREEN}║${NC}"
        echo -e "${GREEN}║${NC} ${RED}⚠ Warnings/Errors: $error_count found${NC}"
    fi
    
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
}

show_completed_experiments() {
    local exp_dir=$1
    
    # Check for results
    if [ -f "${exp_dir}/experiment_summary.json" ]; then
        echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║${NC} ✓ COMPLETED: $(basename "$exp_dir")"
        echo -e "${BLUE}╠══════════════════════════════════════════════════════════════════════╣${NC}"
        
        # Count results
        if [ -d "${exp_dir}/03_results" ]; then
            local csv_count=$(find "${exp_dir}/03_results" -name "*.csv" 2>/dev/null | wc -l)
            local viz_count=$(find "${exp_dir}/03_results/visualizations" -name "*.png" 2>/dev/null | wc -l)
            
            echo -e "${BLUE}║${NC} Results:"
            echo -e "${BLUE}║${NC}   CSV files: $csv_count"
            echo -e "${BLUE}║${NC}   Visualizations: $viz_count"
        fi
        
        echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    fi
}

monitor_experiments() {
    print_header
    
    local found_running=false
    local found_completed=false
    
    # Check for running experiments
    if [ -d "${EXPERIMENTS_DIR}" ]; then
        # First, show running experiments
        for exp_dir in "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/; do
            if [ -d "${exp_dir}/05_logs" ]; then
                for pid_file in "${exp_dir}/05_logs"/*.pid; do
                    if [ -f "$pid_file" ]; then
                        local pid=$(cat "$pid_file" 2>/dev/null)
                        local log_file="${pid_file%.pid}"
                        
                        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                            show_experiment_details "$exp_dir" "$pid" "$log_file"
                            found_running=true
                            echo ""
                        else
                            # Clean up stale PID file
                            rm -f "$pid_file" 2>/dev/null
                        fi
                    fi
                done
            fi
        done
        
        # Show recently completed experiments
        if [ "$found_running" = false ]; then
            echo -e "${YELLOW}No experiments currently running${NC}"
            echo ""
            echo "Recent experiments:"
            
            for exp_dir in $(ls -dt "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/ 2>/dev/null | head -3); do
                if [ -f "${exp_dir}/experiment_summary.json" ]; then
                    show_completed_experiments "$exp_dir"
                    found_completed=true
                    echo ""
                fi
            done
            
            if [ "$found_completed" = false ]; then
                echo "  No completed experiments found"
            fi
        fi
    else
        echo -e "${RED}Experiments directory not found: ${EXPERIMENTS_DIR}${NC}"
    fi
    
    # Show commands
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "Commands:"
    echo "  ${GREEN}q/Ctrl+C${NC}: Quit monitor"
    echo "  ${GREEN}./run_experiment.sh --test${NC}: Run quick test"
    echo "  ${GREEN}./run_experiment.sh --full --background${NC}: Run full experiment"
    echo "  ${GREEN}./run_experiment.sh --stop${NC}: Stop all experiments"
    echo "════════════════════════════════════════════════════════════════════════"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Handle Ctrl+C gracefully
trap 'echo -e "\n\n${YELLOW}Monitoring stopped${NC}"; exit 0' INT TERM

# Check for --once flag
if [ "$1" = "--once" ]; then
    monitor_experiments
else
    # Continuous monitoring loop
    while true; do
        monitor_experiments
        echo ""
        echo -e "${YELLOW}Refreshing in ${REFRESH_INTERVAL} seconds... (Press Ctrl+C to stop)${NC}"
        sleep $REFRESH_INTERVAL
    done
fi