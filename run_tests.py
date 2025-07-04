#!/usr/bin/env python3
"""
Test runner script for the Alpaca trading application.

This script provides a convenient way to run tests from the root directory
with various options for test execution.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False
    
    try:
        import alpaca_trade_api
        print("✅ alpaca-trade-api is available")
    except ImportError:
        print("❌ alpaca-trade-api not found. Install with: pip install alpaca-trade-api")
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib is available")
    except ImportError:
        print("❌ matplotlib not found. Install with: pip install matplotlib")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests for the Alpaca trading application")
    parser.add_argument("--test-file", "-f", help="Specific test file to run (e.g., test_orb.py)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests with verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--lint", "-l", action="store_true", help="Run linting before tests")
    parser.add_argument("--no-deps-check", action="store_true", help="Skip dependency check")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    
    args = parser.parse_args()
    
    # Change to the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"Running tests from: {project_root}")
    
    # Check dependencies unless skipped
    if not args.no_deps_check:
        if not check_dependencies():
            print("\n❌ Dependency check failed. Fix dependencies or use --no-deps-check")
            sys.exit(1)
    
    success = True
    
    # Run linting if requested
    if args.lint:
        lint_cmd = ["flake8", "code/", "atoms/"]
        if not run_command(lint_cmd, "Linting"):
            success = False
    
    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add test file or default to tests directory
    if args.test_file:
        test_path = f"tests/{args.test_file}" if not args.test_file.startswith("tests/") else args.test_file
        pytest_cmd.append(test_path)
    else:
        pytest_cmd.append("tests/")
    
    # Add verbose flag
    if args.verbose:
        pytest_cmd.append("-v")
    
    # Add pattern matching
    if args.pattern:
        pytest_cmd.extend(["-k", args.pattern])
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(["--cov=code", "--cov=atoms", "--cov-report=html", "--cov-report=term"])
    
    # Run pytest
    if not run_command(pytest_cmd, "Running tests"):
        success = False
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All operations completed successfully!")
    else:
        print("❌ Some operations failed. Check the output above.")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()