#!/bin/bash

# =============================================================================
# Perception Prediction Environment Setup Script
# =============================================================================
# This script sets up the Python environment and verifies all dependencies
# for running delta sensitivity experiments
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="perception_env"
PYTHON_VERSION="python3"
MIN_PYTHON_VERSION="3.8"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_python_version() {
    print_step "Checking Python version"

    if ! command -v ${PYTHON_VERSION} &> /dev/null; then
        print_error "Python 3 is not installed"
        echo "Please install Python ${MIN_PYTHON_VERSION} or higher"
        exit 1
    fi

    PYTHON_CMD=$(command -v ${PYTHON_VERSION})
    INSTALLED_VERSION=$(${PYTHON_CMD} --version 2>&1 | awk '{print $2}')
    INSTALLED_MAJOR=$(echo ${INSTALLED_VERSION} | cut -d. -f1)
    INSTALLED_MINOR=$(echo ${INSTALLED_VERSION} | cut -d. -f2)

    MIN_MAJOR=$(echo ${MIN_PYTHON_VERSION} | cut -d. -f1)
    MIN_MINOR=$(echo ${MIN_PYTHON_VERSION} | cut -d. -f2)

    if [ "${INSTALLED_MAJOR}" -lt "${MIN_MAJOR}" ] || \
       ([ "${INSTALLED_MAJOR}" -eq "${MIN_MAJOR}" ] && [ "${INSTALLED_MINOR}" -lt "${MIN_MINOR}" ]); then
        print_error "Python version ${INSTALLED_VERSION} is too old"
        echo "Minimum required: ${MIN_PYTHON_VERSION}"
        exit 1
    fi

    print_success "Python ${INSTALLED_VERSION} detected"
}

create_virtual_environment() {
    print_step "Creating virtual environment"

    cd "${SCRIPT_DIR}"

    if [ -d "${VENV_NAME}" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${VENV_NAME}"
            print_success "Removed existing environment"
        else
            print_warning "Using existing environment"
            return 0
        fi
    fi

    ${PYTHON_VERSION} -m venv "${VENV_NAME}"
    print_success "Virtual environment created: ${VENV_NAME}"
}

activate_environment() {
    print_step "Activating virtual environment"

    source "${SCRIPT_DIR}/${VENV_NAME}/bin/activate"

    print_success "Virtual environment activated"
    echo "Python: $(which python3)"
    echo "Version: $(python3 --version)"
}

upgrade_pip() {
    print_step "Upgrading pip"

    pip install --upgrade pip setuptools wheel > /dev/null 2>&1

    print_success "pip upgraded to $(pip --version | awk '{print $2}')"
}

install_dependencies() {
    print_step "Installing dependencies"

    if [ ! -f "${SCRIPT_DIR}/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi

    echo "Installing packages (this may take a few minutes)..."
    pip install -r "${SCRIPT_DIR}/requirements.txt"

    print_success "All dependencies installed"
}

verify_imports() {
    print_step "Verifying package imports"

    python3 << 'EOF'
import sys

packages = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "matplotlib",
    "seaborn",
    "xgboost",
    "openpyxl",
    "yaml",
    "tqdm",
    "joblib",
    "psutil"
]

failed = []
for package in packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError as e:
        print(f"  ✗ {package}: {e}")
        failed.append(package)

if failed:
    print(f"\nFailed to import: {', '.join(failed)}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "All imports verified"
    else
        print_error "Some imports failed"
        exit 1
    fi
}

verify_data() {
    print_step "Verifying data files"

    DATA_DIR="${SCRIPT_DIR}/Input_Data/dinov3_all_classes"

    if [ ! -d "${DATA_DIR}" ]; then
        print_warning "Data directory not found: ${DATA_DIR}"
        echo "Please ensure your data is in the correct location"
        return 1
    fi

    REQUIRED_FILES=("beautiful_input.xlsx" "lively_input.xlsx" "boring_input.xlsx" "safe_input.xlsx")

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "${DATA_DIR}/${file}" ]; then
            SIZE=$(du -h "${DATA_DIR}/${file}" | cut -f1)
            print_success "${file} (${SIZE})"
        else
            print_warning "${file} not found"
        fi
    done
}

verify_scripts() {
    print_step "Verifying core scripts"

    CORE_SCRIPTS=(
        "run_experiment.sh"
        "monitor_experiment.sh"
        "core_scripts/enhanced_experiment_runner.py"
        "multiclass_delta_sensitivity.py"
        "multiclass_evaluator.py"
    )

    for script in "${CORE_SCRIPTS[@]}"; do
        if [ -f "${SCRIPT_DIR}/${script}" ]; then
            print_success "${script}"
        else
            print_error "${script} not found"
            exit 1
        fi
    done

    # Make shell scripts executable
    chmod +x "${SCRIPT_DIR}"/*.sh 2>/dev/null || true
}

create_directories() {
    print_step "Creating required directories"

    mkdir -p "${SCRIPT_DIR}/experiments"
    print_success "experiments/ directory created"
}

print_summary() {
    print_header "SETUP COMPLETE"

    echo ""
    echo "Environment Details:"
    echo "  Location: ${SCRIPT_DIR}/${VENV_NAME}"
    echo "  Python: $(python3 --version)"
    echo "  Pip: $(pip --version | awk '{print $2}')"
    echo ""
    echo "Next Steps:"
    echo "  1. Activate environment:"
    echo "     source ${VENV_NAME}/bin/activate"
    echo ""
    echo "  2. Run quick test (5-15 min):"
    echo "     ./run_experiment.sh --test"
    echo ""
    echo "  3. Run full experiment (2-6 hours):"
    echo "     ./run_experiment.sh --full"
    echo ""
    echo "  4. Run in background:"
    echo "     ./run_experiment.sh --full --background"
    echo ""
    echo "  5. Monitor progress:"
    echo "     ./monitor_experiment.sh"
    echo ""
    echo "For help:"
    echo "  ./run_experiment.sh --help"
    echo ""
    print_success "Ready to run experiments!"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print_header "PERCEPTION PREDICTION ENVIRONMENT SETUP"

check_python_version
create_virtual_environment
activate_environment
upgrade_pip
install_dependencies
verify_imports
verify_data
verify_scripts
create_directories
print_summary

echo ""
echo "========================================================================"
echo "✓ Setup completed successfully!"
echo "========================================================================"
echo ""
