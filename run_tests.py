#!/usr/bin/env python
"""
CRISPY Modern Test Runner 🚀

Convenient wrapper for running pytest with commonly used options and emojis!
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(args):
    """Run pytest with the given arguments"""
    cmd = ["pytest"] + args
    print(f"🏃‍♂️ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="🧪 CRISPY Modern Test Runner with Beautiful Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Unit tests only  
  python run_tests.py --working          # Working tests only
  python run_tests.py --fast             # Fast tests (no slow ones)
  python run_tests.py --coverage         # With coverage report
  python run_tests.py --parallel         # Run in parallel
  python run_tests.py --experimental     # Include experimental tests
        """
    )
    
    # Test selection options
    parser.add_argument("--unit", action="store_true", help="🧪 Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="🚀 Run integration tests only") 
    parser.add_argument("--working", action="store_true", help="✅ Run only working tests")
    parser.add_argument("--experimental", action="store_true", help="⚠️ Include experimental tests")
    parser.add_argument("--fast", action="store_true", help="⚡ Skip slow tests")
    parser.add_argument("--data", action="store_true", help="📁 Only tests requiring reference data")
    
    # Output options
    parser.add_argument("--coverage", action="store_true", help="📊 Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="🏃‍♂️ Run tests in parallel")
    parser.add_argument("--quiet", action="store_true", help="🤫 Quiet output")
    parser.add_argument("--verbose", action="store_true", help="📝 Verbose output")
    
    # Test files
    parser.add_argument("files", nargs="*", help="🎯 Specific test files to run")
    
    args = parser.parse_args()
    
    # Build pytest command
    pytest_args = []
    
    # Test selection
    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"]) 
    elif args.working:
        pytest_args.extend(["-m", "working"])
    elif args.data:
        pytest_args.extend(["-m", "requires_data"])
    
    # Experimental tests (usually excluded by default)
    if not args.experimental:
        if any(m in pytest_args for m in ["-m"]):
            # Already have a marker, combine with 'and not experimental'
            idx = pytest_args.index("-m") + 1
            pytest_args[idx] = pytest_args[idx] + " and not experimental"
        else:
            pytest_args.extend(["-m", "not experimental"])
    
    # Speed options
    if args.fast:
        if any(m in pytest_args for m in ["-m"]):
            idx = pytest_args.index("-m") + 1
            pytest_args[idx] = pytest_args[idx] + " and not slow"
        else:
            pytest_args.extend(["-m", "not slow"])
    
    # Output options
    if args.coverage:
        pytest_args.extend(["--cov=crispy", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        pytest_args.extend(["-n", "auto"])
    
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")
    
    # Add specific files
    if args.files:
        pytest_args.extend(args.files)
    
    # Print header
    print("🎯 CRISPY Modern Test Runner")
    print("=" * 50)
    
    # Run pytest
    exit_code = run_pytest(pytest_args)
    
    # Print footer
    print("=" * 50)
    if exit_code == 0:
        print("🎉 Testing completed successfully!")
    else:
        print("⚠️ Some tests failed or had issues")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)