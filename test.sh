#!/bin/bash
# Simple test runner script for the Alpaca trading application

set -e  # Exit on any error

echo "=========================================="
echo "Running Alpaca Trading Application Tests"
echo "=========================================="

# Check if we're in the right directory
if [[ ! -f "code/orb.py" ]]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Function to run tests with different options
run_tests() {
    echo "Running tests..."
    if [[ "$1" == "verbose" ]]; then
        python -m pytest tests/ -v
    elif [[ "$1" == "coverage" ]]; then
        python -m pytest tests/ --cov=code --cov=atoms --cov-report=html --cov-report=term
    elif [[ "$1" == "specific" ]]; then
        python -m pytest tests/"$2" -v
    else
        python -m pytest tests/
    fi
}

# Parse command line arguments
case "${1:-}" in
    "lint")
        echo "Running linting..."
        flake8 code/ atoms/ || echo "Linting completed with warnings"
        run_tests
        ;;
    "verbose"|"-v")
        run_tests verbose
        ;;
    "coverage"|"-c")
        run_tests coverage
        ;;
    "orb"|"test_orb")
        run_tests specific "test_orb.py"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo "Options:"
        echo "  (no args)    - Run all tests"
        echo "  verbose|-v   - Run tests with verbose output"
        echo "  coverage|-c  - Run tests with coverage report"
        echo "  lint         - Run linting then tests"
        echo "  orb          - Run only ORB tests"
        echo "  help|-h      - Show this help message"
        exit 0
        ;;
    "")
        run_tests
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo "=========================================="
echo "Tests completed successfully!"
echo "=========================================="