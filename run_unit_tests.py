#!/usr/bin/env python
"""
CRISPY Unit Test Runner

⚠️  DEPRECATED: This legacy test runner has been replaced by pytest.
This script now redirects to the modern pytest framework for better reliability.

For direct pytest usage:
  pytest                    # Run all tests
  pytest -m unit           # Unit tests only  
  pytest -m working        # Working tests only
  pytest --cov=crispy      # With coverage

For more options, use the modern test runners:
  python run_tests.py --help
  python run_simple_tests.py
"""

import sys
import subprocess

def main():
    print("🔄 CRISPY Unit Test Runner (Legacy)")
    print("=" * 50)
    print("⚠️  This legacy runner has been replaced by pytest.")
    print("🚀 Redirecting to modern pytest framework...")
    print()
    
    # Run pytest with unit test focus
    pytest_args = [
        "pytest", 
        "-v",                    # Verbose output
        "-m", "unit",           # Unit tests only
        "--tb=short"            # Shorter traceback format
    ]
    
    print(f"Running: {' '.join(pytest_args)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(pytest_args)
        return result.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("   conda install pytest")
        print("   # or")
        print("   pip install pytest")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)