#!/bin/bash

# =============================================================================
# Enhanced Delta Sensitivity Experiment Runner
# =============================================================================
# This script sets up the environment and runs the enhanced delta sensitivity 
# analysis experiment with publication-ready visualizations
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory (auto-detect)
if [ -z "${BASE_DIR}" ]; then
    export BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Virtual environment
VENV_NAME="perception_env"
# Use relative path from base directory
VENV_PATH="${BASE_DIR}/${VENV_NAME}"
PYTHON_VERSION="python3"

# Paths
INPUT_DATA_DIR="${BASE_DIR}/Input_Data/dinov3_all_classes"
EXPERIMENTS_DIR="${BASE_DIR}/experiments"
CORE_SCRIPTS_DIR="${BASE_DIR}/core_scripts"

# Experiment subdirectory structure (no flags; simple config here)
# Derive dataset name from INPUT_DATA_DIR and allow a fixed TEST_CITY here
DATASET_NAME="$(basename "${INPUT_DATA_DIR}")"
TEST_CITY_NAME="${TEST_CITY_NAME:-Mumbai}"
# Nested structure: experiments/<dataset>/<test_city>/
export EXPERIMENTS_SUBDIR="${DATASET_NAME}/${TEST_CITY_NAME}"

# Cities config for validation
CITIES_CONFIG="${BASE_DIR}/config/cities.yaml"

# Feature columns count (default 36)
FEATURE_COUNT="${FEATURE_COUNT:-36}"
export FEATURE_COUNT

# Test set configuration (simple: set city here, no CLI needed)
TEST_CITIES="${TEST_CITY_NAME}"  # Pass single city from TEST_CITY_NAME to runner
USE_CITY_BASED_SPLIT="${USE_CITY_BASED_SPLIT:-true}"  # Use city-based split by default

# NOTE: Python script will create experiment directory with timestamp
# No need to create EXPERIMENT_NAME or EXPERIMENT_DIR here

# =============================================================================
# FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

print_step() {
    echo ""
    echo ">>> $1"
}

check_requirements() {
    print_header "CHECKING REQUIREMENTS"
    
    # Check Python
    if ! command -v ${PYTHON_VERSION} &> /dev/null; then
        echo "ERROR: ${PYTHON_VERSION} is not installed"
        exit 1
    fi
    
    # Check input data
    if [ ! -d "${INPUT_DATA_DIR}" ]; then
        echo "ERROR: Input data directory not found: ${INPUT_DATA_DIR}"
        exit 1
    fi
    
    # Check required files
    REQUIRED_FILES=("beautiful_input.xlsx" "lively_input.xlsx" "boring_input.xlsx" "safe_input.xlsx")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${INPUT_DATA_DIR}/${file}" ]; then
            echo "ERROR: Required file not found: ${INPUT_DATA_DIR}/${file}"
            exit 1
        fi
    done
    
    # Check core scripts
    if [ ! -d "${CORE_SCRIPTS_DIR}" ]; then
        echo "ERROR: Core scripts directory not found: ${CORE_SCRIPTS_DIR}"
        exit 1
    fi
    
    # Validate city name against config if available
    if [ -f "${CITIES_CONFIG}" ]; then
        if ! command -v yq >/dev/null 2>&1; then
            echo "WARNING: 'yq' not found; skipping city name validation against ${CITIES_CONFIG}"
        else
            VALID_CITIES=$(yq '.cities[]' "${CITIES_CONFIG}" 2>/dev/null | tr -d '"')
            if ! echo "${VALID_CITIES}" | grep -qx "${TEST_CITY_NAME}"; then
                echo "WARNING: TEST_CITY_NAME='${TEST_CITY_NAME}' not found in ${CITIES_CONFIG}."
                echo "         Valid cities are: ${VALID_CITIES}" | tr '\n' ' ' && echo ""
                echo "         Proceeding anyway."
            fi
        fi
    fi
    
    echo "✓ All requirements satisfied"
    echo "✓ Python version: $(${PYTHON_VERSION} --version)"
    echo "✓ Input data: ${INPUT_DATA_DIR}"
    echo "✓ Core scripts: ${CORE_SCRIPTS_DIR}"
}

activate_environment() {
    print_header "ACTIVATING VIRTUAL ENVIRONMENT"
    
    if [ ! -d "${VENV_PATH}" ]; then
        echo "ERROR: Virtual environment not found at ${VENV_PATH}"
        echo "Please run: source setup_experiment.sh first"
        exit 1
    fi
    
    source "${VENV_PATH}/bin/activate"
    
    echo "✓ Virtual environment activated"
    echo "✓ Python: $(which python3)"
    echo "✓ Using: $(python3 --version)"
}

check_for_running_experiments() {
    if [ -d "${EXPERIMENTS_DIR}" ]; then
        for exp_dir in "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/; do
            if [ -d "${exp_dir}/05_logs" ]; then
                for pid_file in "${exp_dir}/05_logs"/*.pid; do
                    if [ -f "$pid_file" ]; then
                        local pid=$(cat "$pid_file" 2>/dev/null)
                        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                            echo "⚠ WARNING: Experiment already running with PID: $pid"
                            echo "  Experiment: $(basename $(dirname $(dirname "$pid_file")))"
                            echo ""
                            echo "Options:"
                            echo "  1. Monitor progress: ./monitor_experiment.sh"
                            echo "  2. Stop current: ./run_experiment.sh --stop"
                            echo "  3. Check status: ./run_experiment.sh --status"
                            echo ""
                            read -p "Stop running experiment and start new? (y/N): " -n 1 -r
                            echo
                            if [[ $REPLY =~ ^[Yy]$ ]]; then
                                echo "Stopping PID $pid..."
                                kill "$pid" 2>/dev/null
                                sleep 2
                                kill -9 "$pid" 2>/dev/null || true
                                rm -f "$pid_file"
                                echo "✓ Previous experiment stopped"
                            else
                                echo "Aborting. Current experiment continues."
                                return 1
                            fi
                        else
                            rm -f "$pid_file"  # Clean stale PID
                        fi
                    fi
                done
            fi
        done
    fi
    return 0
}

run_experiment() {
    print_header "RUNNING ENHANCED EXPERIMENT"
    
    # Check for running experiments
    if ! check_for_running_experiments; then
        return 1
    fi
    
    # Export paths for Python scripts
    export INPUT_DATA_DIR
    export BASE_DIR
    export TEST_CITIES
    
    print_step "Starting delta sensitivity experiment"
    print_step "Test cities: ${TEST_CITIES}"
    print_step "Using city-based split: ${USE_CITY_BASED_SPLIT}"
    
    # Python script will create experiment directory with timestamp
    # Use a simple temporary log file in base directory
    TEMP_LOG_FILE="${BASE_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"
    
    if [ "${RUN_IN_BACKGROUND:-false}" = "true" ]; then
        print_step "Running in background mode"
        echo "Temporary log: ${TEMP_LOG_FILE}"
        
        # Use the original runner script for both methods
        RUNNER_SCRIPT="${CORE_SCRIPTS_DIR}/enhanced_experiment_runner.py"
        
        RUNNER_ARGS="--base-dir ${BASE_DIR} --test-cities ${TEST_CITY_NAME} ${REALMLP_FLAG:+--realmlp ${REALMLP_FLAG}} ${CPU_ONLY:+--cpu-only}"
        
        # Run in background with nohup (use system python3 since venv is broken)
        nohup python3 "${RUNNER_SCRIPT}" ${RUNNER_ARGS} \
            > "${TEMP_LOG_FILE}" 2>&1 &
        
        EXPERIMENT_PID=$!
        
        echo "✓ Experiment started with PID: $EXPERIMENT_PID"
        echo ""
        echo "Monitor progress:"
        echo "  ./run_experiment.sh --status"
        echo "  tail -f ${TEMP_LOG_FILE}"
        echo ""
        echo "Stop experiment:"
        echo "  kill $EXPERIMENT_PID"
        echo ""
        echo "💡 You can now close this terminal safely"
        echo "💡 Log will be moved to proper experiment directory automatically"
        
    else
        # Use the original runner script for both methods
        RUNNER_SCRIPT="${CORE_SCRIPTS_DIR}/enhanced_experiment_runner.py"
        
        RUNNER_ARGS="--base-dir ${BASE_DIR} --test-cities ${TEST_CITY_NAME} ${REALMLP_FLAG:+--realmlp ${REALMLP_FLAG}} ${CPU_ONLY:+--cpu-only}"
        
        # Run in foreground
        python3 "${RUNNER_SCRIPT}" ${RUNNER_ARGS} \
            2>&1 | tee "${TEMP_LOG_FILE}"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Experiment completed successfully!"
            # Find the latest experiment directory
            LATEST_EXP_DIR=$(ls -dt "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/ 2>/dev/null | head -1)
            if [ -n "${LATEST_EXP_DIR}" ]; then
                echo "Results: ${LATEST_EXP_DIR}/03_results/"
            fi
        else
            echo "✗ Experiment failed. Check logs: ${TEMP_LOG_FILE}"
            exit 1
        fi
    fi
}

run_quick_test() {
    print_header "RUNNING QUICK TEST"
    
    # Check for running experiments
    if ! check_for_running_experiments; then
        return 1
    fi
    
    export INPUT_DATA_DIR
    export BASE_DIR
    export TEST_CITIES
    export USE_CITY_BASED_SPLIT
    
    print_step "Starting quick test (beautiful perception, 2 deltas, 1 model)"
    print_step "Test city: ${TEST_CITY_NAME}"
    
    # Python script will create experiment directory with timestamp
    # Use a simple temporary log file in base directory
    TEMP_LOG_FILE="${BASE_DIR}/quick_test_$(date +%Y%m%d_%H%M%S).log"
    
    if [ "${RUN_IN_BACKGROUND:-false}" = "true" ]; then
        print_step "Running test in background"
        
        # Unified runner only
        RUNNER_SCRIPT="${CORE_SCRIPTS_DIR}/enhanced_experiment_runner.py"
        RUNNER_ARGS="--base-dir ${BASE_DIR} --test-cities ${TEST_CITY_NAME} --test"
        
        nohup python3 "${RUNNER_SCRIPT}" ${RUNNER_ARGS} \
            > "${TEMP_LOG_FILE}" 2>&1 &
        
        TEST_PID=$!
        
        echo "✓ Quick test started with PID: $TEST_PID"
        echo "Monitor: tail -f ${TEMP_LOG_FILE}"
        
    else
        # Unified runner only
        RUNNER_SCRIPT="${CORE_SCRIPTS_DIR}/enhanced_experiment_runner.py"
        RUNNER_ARGS="--base-dir ${BASE_DIR} --test-cities ${TEST_CITY_NAME} --test"
        
        python3 "${RUNNER_SCRIPT}" ${RUNNER_ARGS} 2>&1 | tee "${TEMP_LOG_FILE}"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Quick test completed!"
        else
            echo "✗ Quick test failed"
            exit 1
        fi
    fi
}

check_status() {
    print_header "EXPERIMENT STATUS"
    
    local found_running=false
    
    # First check for PID files in experiment directories
    if [ -d "${EXPERIMENTS_DIR}" ]; then
        for exp_dir in "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/; do
            if [ -d "${exp_dir}/05_logs" ]; then
                local exp_name=$(basename "$exp_dir")
                local has_running=false
                
                for pid_file in "${exp_dir}/05_logs"/*.pid; do
                    if [ -f "$pid_file" ]; then
                        local pid=$(cat "$pid_file")
                        local log_file="${pid_file%.pid}"
                        
                        if kill -0 "$pid" 2>/dev/null; then
                            if [ "$has_running" = false ]; then
                                echo "📊 Experiment: $exp_name"
                                has_running=true
                            fi
                            echo "  ✓ Running: PID $pid"
                            echo "    Log: $(basename "$log_file")"
                            echo "    Runtime: $(ps -o etime= -p $pid 2>/dev/null | tr -d ' ')"
                            
                            # Show detailed progress
                            if [ -f "$log_file" ]; then
                                echo ""
                                echo "  📈 DETAILED PROGRESS:"
                                show_experiment_progress "$log_file" "$pid"
                            fi
                            
                            found_running=true
                        else
                            rm -f "$pid_file"  # Clean stale PID
                        fi
                    fi
                done
                
                if [ "$has_running" = true ]; then
                    echo ""
                fi
            fi
        done
    fi
    
    # Fallback: Look for running processes by name if no PID files found
    if [ "$found_running" = false ]; then
        local running_pids=$(ps aux | grep "enhanced_experiment_runner.py" | grep -v grep | awk '{print $2}')
        
        if [ -n "$running_pids" ]; then
            echo "🔍 Found running experiments (by process detection):"
            echo ""
            
            for pid in $running_pids; do
                local cmd_line=$(ps -p "$pid" -o args= 2>/dev/null)
                local runtime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
                
                echo "  ✓ Running: PID $pid"
                echo "    Command: enhanced_experiment_runner.py"
                echo "    Runtime: $runtime"
                
                # Try to find the corresponding log file
                local temp_log_file=$(ls -t "${BASE_DIR}"/experiment_*.log 2>/dev/null | head -1)
                if [ -f "$temp_log_file" ]; then
                    echo "    Log: $(basename "$temp_log_file")"
                    echo ""
                    echo "  📈 DETAILED PROGRESS:"
                    show_experiment_progress "$temp_log_file" "$pid"
                else
                    # Try to find experiment directory log
                    local latest_exp_dir=$(ls -dt "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/ 2>/dev/null | head -1)
                    if [ -n "$latest_exp_dir" ] && [ -f "${latest_exp_dir}/05_logs/"*.log ]; then
                        local exp_log=$(ls -t "${latest_exp_dir}/05_logs/"*.log 2>/dev/null | head -1)
                        echo "    Log: $(basename "$exp_log")"
                        echo ""
                        echo "  📈 DETAILED PROGRESS:"
                        show_experiment_progress "$exp_log" "$pid"
                    fi
                fi
                echo ""
                
                found_running=true
            done
        fi
    fi
    
    if [ "$found_running" = false ]; then
        echo "No experiments currently running"
        
        # Show recent experiments
        if [ -d "${EXPERIMENTS_DIR}" ]; then
            echo ""
            echo "Recent experiments:"
            ls -dt "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/ 2>/dev/null | head -3 | while read dir; do
                echo "  - $(basename "$dir")"
                if [ -f "${dir}/experiment_summary.json" ]; then
                    echo "    ✓ Completed with results"
                fi
            done
        fi
    fi
}

show_experiment_progress() {
    local log_file="$1"
    local pid="$2"
    
    # Get last 50 lines to analyze recent progress
    local recent_logs=$(tail -n 50 "$log_file" 2>/dev/null)
    
    # Detect experiment configuration from log
    local total_experiments=$(echo "$recent_logs" | grep -o "Total experiments to run: [0-9]*" | tail -1 | sed 's/Total experiments to run: //')
    
    # Default values if not found in log
    if [ -z "$total_experiments" ]; then
        # Try to get from entire log file
        total_experiments=$(grep -o "Total experiments to run: [0-9]*" "$log_file" 2>/dev/null | tail -1 | sed 's/Total experiments to run: //')
        if [ -z "$total_experiments" ]; then
            # Fallback to full experiment size
            total_experiments=112
        fi
    fi
    
    local total_combinations=$total_experiments
    
    # Extract current perception being processed
    local current_perception=$(echo "$recent_logs" | grep -o "Processing Perception: [A-Z]*" | tail -1 | sed 's/Processing Perception: //' | tr '[:upper:]' '[:lower:]')
    
    # Extract current delta being processed  
    local current_delta=$(echo "$recent_logs" | grep -o "Processing delta=[0-9.]*" | tail -1 | sed 's/Processing delta=//')
    
    # Extract current model being trained
    local current_model=$(echo "$recent_logs" | grep -o "Training [a-z_]* for" | tail -1 | sed 's/Training //' | sed 's/ for//')
    
    # Count completed models (search entire log file)
    local completed_models=$(grep -E "Model.*trained in [0-9.]*s" "$log_file" 2>/dev/null | wc -l || echo 0)
    local completed_perceptions=$(grep -E "Completed perception:" "$log_file" 2>/dev/null | wc -l || echo 0)
    
    # Calculate progress percentage
    local progress_percent=0
    if [ "$total_combinations" -gt 0 ] && [ "$completed_models" -gt 0 ]; then
        progress_percent=$(( (completed_models * 100) / total_combinations ))
    fi
    
    # Show current status with better detection
    if [ -n "$current_perception" ] && [ -n "$current_delta" ] && [ -n "$current_model" ]; then
        echo "    🔄 Currently: $current_perception (δ=$current_delta) → $current_model"
    elif [ -n "$current_perception" ] && [ -n "$current_delta" ]; then
        echo "    🔄 Currently: Processing $current_perception (δ=$current_delta)"
    elif [ -n "$current_perception" ]; then
        echo "    🔄 Currently: Processing perception '$current_perception'"
    else
        # Check various phases based on recent log messages
        if echo "$recent_logs" | grep -q "Loading data from:"; then
            echo "    📁 Currently: Loading and preprocessing data"
        elif echo "$recent_logs" | grep -q "Place Pulse pool:"; then
            echo "    📊 Currently: Splitting data (Place Pulse vs Local test)"
        elif echo "$recent_logs" | grep -q "Generating.*visualizations"; then
            echo "    🎨 Currently: Generating publication-ready visualizations"
        elif echo "$recent_logs" | grep -q "Generating.*report"; then
            echo "    📝 Currently: Generating comprehensive analysis reports"
        elif echo "$recent_logs" | grep -q "EXPERIMENT COMPLETED"; then
            echo "    ✅ Currently: Experiment completed, generating final outputs"
        else
            echo "    ⏳ Currently: Initializing experiment..."
        fi
    fi
    
    # Show progress bar
    echo "    📊 Progress: $completed_models/$total_combinations models completed ($progress_percent%)"
    
    # Create simple progress bar
    local bar_length=30
    local filled_length=$(( (progress_percent * bar_length) / 100 ))
    local bar="    ["
    for ((i=0; i<filled_length; i++)); do bar+="="; done
    for ((i=filled_length; i<bar_length; i++)); do bar+=" "; done
    bar+="]"
    echo "$bar $progress_percent%"
    
    # Show completed perceptions (estimate based on models completed)
    local estimated_perceptions_completed=$(( completed_models / 7 ))
    if [ "$estimated_perceptions_completed" -gt 0 ]; then
        # Get actual perception count from log if available
        local perception_list=$(grep -o "Perceptions: \[.*\]" "$log_file" 2>/dev/null | tail -1)
        local total_perceptions=4  # Default
        if [[ "$perception_list" == *"'beautiful'"* ]]; then
            local perception_count=$(echo "$perception_list" | grep -o "'" | wc -l)
            total_perceptions=$(( perception_count / 2 ))
        fi
        echo "    ✅ Completed perceptions: $completed_perceptions/$total_perceptions (estimated)"
    fi
    
    # Show recent activity (last 3 lines of actual progress) 
    echo ""
    echo "    📝 Recent activity:"
    echo "$recent_logs" | grep -E "(Processing|Training|trained|Completed|Loading|Place Pulse|ERROR|Failed|Generating)" | tail -3 | sed 's/.*INFO - //' | sed 's/^/       /'
    
    # Check for errors
    local error_count=$(grep -E "ERROR|Failed|Exception" "$log_file" 2>/dev/null | wc -l || echo 0)
    if [ "$error_count" -gt 0 ]; then
        echo ""
        echo "    ⚠️  Warnings: $error_count errors/warnings found in log"
        echo "       Latest error:"
        grep "ERROR\|Failed\|Exception" "$log_file" | tail -1 | sed 's/^/       /'
    fi
    
    # Estimate time remaining
    if [ "$progress_percent" -gt 5 ]; then
        local runtime_seconds=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -n "$runtime_seconds" ] && [ "$runtime_seconds" -gt 0 ]; then
            local total_estimated_seconds=$(( (runtime_seconds * 100) / progress_percent ))
            local remaining_seconds=$(( total_estimated_seconds - runtime_seconds ))
            local remaining_minutes=$(( remaining_seconds / 60 ))
            
            if [ "$remaining_minutes" -gt 60 ]; then
                local remaining_hours=$(( remaining_minutes / 60 ))
                local remaining_mins=$(( remaining_minutes % 60 ))
                echo "    ⏱️  Estimated time remaining: ~${remaining_hours}h ${remaining_mins}m"
            else
                echo "    ⏱️  Estimated time remaining: ~${remaining_minutes}m"
            fi
        fi
    fi
}

stop_experiments() {
    print_header "STOPPING EXPERIMENTS"
    
    local stopped_count=0
    
    # First try to stop experiments using PID files
    if [ -d "${EXPERIMENTS_DIR}" ]; then
        for exp_dir in "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/; do
            if [ -d "${exp_dir}/05_logs" ]; then
                for pid_file in "${exp_dir}/05_logs"/*.pid; do
                    if [ -f "$pid_file" ]; then
                        local pid=$(cat "$pid_file")
                        if kill -0 "$pid" 2>/dev/null; then
                            echo "Stopping PID $pid ($(basename $(dirname $(dirname "$pid_file"))))"
                            kill "$pid"
                            sleep 2
                            kill -9 "$pid" 2>/dev/null || true
                            rm -f "$pid_file"
                            ((stopped_count++))
                        else
                            rm -f "$pid_file"
                        fi
                    fi
                done
            fi
        done
    fi
    
    # Fallback: Look for running processes by name if no PID files found
    if [ $stopped_count -eq 0 ]; then
        local running_pids=$(ps aux | grep "enhanced_experiment_runner.py" | grep -v grep | awk '{print $2}')
        
        if [ -n "$running_pids" ]; then
            echo "🔍 Found running experiments by process detection:"
            
            for pid in $running_pids; do
                local runtime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
                echo "Stopping PID $pid (runtime: $runtime)"
                
                # Graceful termination first
                kill "$pid" 2>/dev/null || true
                sleep 3
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    echo "  Force killing PID $pid..."
                    kill -9 "$pid" 2>/dev/null || true
                fi
                
                ((stopped_count++))
            done
        fi
    fi
    
    if [ $stopped_count -eq 0 ]; then
        echo "No running experiments found"
    else
        echo "✓ Stopped $stopped_count experiment(s)"
    fi
}

show_results() {
    print_header "EXPERIMENT RESULTS"
    
    # Find the most recent experiment directory
    local latest_exp_dir=$(ls -dt "${EXPERIMENTS_DIR}"/perception_delta_sensitivity_*/ 2>/dev/null | head -1)
    
    if [ -n "${latest_exp_dir}" ] && [ -d "${latest_exp_dir}/03_results" ]; then
        echo "Results directory: ${latest_exp_dir}/03_results/"
        echo ""
        
        # Check for key files
        if [ -f "${latest_exp_dir}/03_results/metrics/all_results.csv" ]; then
            echo "✓ Metrics: all_results.csv"
        fi
        
        if [ -d "${latest_exp_dir}/03_results/visualizations" ]; then
            local viz_count=$(find "${latest_exp_dir}/03_results/visualizations" -name "*.png" 2>/dev/null | wc -l)
            echo "✓ Visualizations: $viz_count figures generated"
        fi
        
        if [ -f "${latest_exp_dir}/experiment_summary.json" ]; then
            echo "✓ Summary: experiment_summary.json"
        fi
    else
        echo "No results found yet"
    fi
}

print_usage() {
    echo "Usage: $0 [OPTION] [--background]"
    echo ""
    echo "Options:"
    echo "  --test       Run quick test (1 perception, 2 deltas, 1 model)"
    echo "  --full       Run complete experiment (default)"
    echo "  --cpu-only   Force CPU-only execution (sets CUDA_VISIBLE_DEVICES=)"
    echo "  --realmlp {td|hpo|both|off}  Select RealMLP variant (default: td)"
    echo "  --status     Check status of running experiments"
    echo "  --stop       Stop all running experiments"
    echo "  --help       Show this help message"
    echo ""
    echo "Test set configuration:"
    echo "  --test-cities CITY1,CITY2,...  Specify test cities (default: Mumbai)"
    echo "  --use-last-280                Use last 280 entries instead of city-based split"
    echo ""
    echo "Background execution:"
    echo "  --background Run in background (survives terminal close)"
    echo ""
    echo "Examples:"
    echo "  $0 --test                                    # Quick test with Mumbai"
    echo "  $0 --test --test-cities Mumbai,Tokyo        # Quick test with Mumbai and Tokyo"
    echo "  $0 --full --test-cities Berlin,Paris        # Full experiment with Berlin and Paris"
    echo "  $0 --full --use-last-280                    # Use original last 280 entries method"
    echo "  $0 --status                                  # Check running experiments"
    echo ""
    echo "Configuration:"
    echo "  Base directory: ${BASE_DIR}"
    echo "  Virtual env: ${VENV_PATH}"
    echo "  Test cities: ${TEST_CITIES}"
    echo "  City-based split: ${USE_CITY_BASED_SPLIT}"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print_header "ENHANCED DELTA SENSITIVITY EXPERIMENT"
echo "Base directory: ${BASE_DIR}"
echo "Started at: $(date)"

# Parse arguments
RUN_IN_BACKGROUND=false
COMMAND="${1:---full}"
CPU_ONLY=false
REALMLP_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --background)
            RUN_IN_BACKGROUND=true
            export RUN_IN_BACKGROUND
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --use-last-280)
            USE_CITY_BASED_SPLIT=false
            export USE_CITY_BASED_SPLIT
            shift
            ;;
        --test-cities)
            TEST_CITIES="$2"
            export TEST_CITIES
            shift 2
            ;;
        --test-cities=*)
            TEST_CITIES="${1#*=}"
            export TEST_CITIES
            shift
            ;;
        --realmlp)
            REALMLP_FLAG="$2"
            shift 2
            ;;
        --realmlp=*)
            REALMLP_FLAG="${1#*=}"
            shift
            ;;
        --test|--full|--status|--stop|--help)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Apply CPU-only env if requested
if [ "$CPU_ONLY" = true ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

# Execute command
case "${COMMAND}" in
    --test)
        check_requirements
        activate_environment
        run_quick_test
        [ "${RUN_IN_BACKGROUND}" = "false" ] && show_results
        ;;
    --full)
        check_requirements
        activate_environment
        run_experiment
        [ "${RUN_IN_BACKGROUND}" = "false" ] && show_results
        ;;
    --status)
        check_status
        ;;
    --stop)
        stop_experiments
        ;;
    --help)
        print_usage
        ;;
    *)
        echo "Unknown option: $COMMAND"
        print_usage
        exit 1
        ;;
esac

print_header "COMPLETED"
echo "Finished at: $(date)"
echo "✓ Done!"