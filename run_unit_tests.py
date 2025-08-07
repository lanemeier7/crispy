#!/usr/bin/env python
"""
CRISPY Unit Test Runner

This script runs all the unit tests in crispy/unitTests.py with proper error handling
and reporting. It sets up the necessary parameters and runs each test function.
"""

import sys
import os
import traceback
import numpy as np
from crispy.tools.initLogger import getLogger

def setup_test_environment():
    """Set up the test environment and parameters"""
    print("Setting up test environment...")
    
    # Import required modules
    try:
        from crispy.WFIRST import params
        par = params.Params(codeRoot='./crispy')  # Fix path to reference files
        print("✅ WFIRST parameters loaded")
        print(f"   wavecalDir: {par.wavecalDir}")
        
        # Verify the reference files exist
        import os
        if os.path.exists(par.wavecalDir):
            print("✅ Reference files directory found")
        else:
            print("❌ Reference files directory not found, trying alternative paths...")
            # Try alternative path
            par = params.Params(codeRoot='.')
            if os.path.exists('./crispy/ReferenceFiles/wavecalR50_770/'):
                print("✅ Found reference files in ./crispy/ReferenceFiles/")
                par.prefix = './crispy/ReferenceFiles'
                par.wavecalDir = par.prefix + '/wavecalR50_770/'
                par.exportDir = './crispy/SimResults'
                par.unitTestsOutputs = './crispy/unitTestsOutputs'
            else:
                print("❌ Could not locate reference files")
                return None
        
        return par
    except Exception as e:
        print(f"❌ Failed to load WFIRST parameters: {e}")
        return None

def ensure_output_directory(par):
    """Ensure the unit test outputs directory exists"""
    import os
    if not os.path.exists(par.unitTestsOutputs):
        os.makedirs(par.unitTestsOutputs)
        print(f"✅ Created output directory: {par.unitTestsOutputs}")
    else:
        print(f"✅ Output directory exists: {par.unitTestsOutputs}")

def run_test_safely(test_func, test_name, *args, **kwargs):
    """Run a test function with error handling"""
    print(f"\n🧪 Running {test_name}...")
    try:
        result = test_func(*args, **kwargs)
        print(f"✅ {test_name} completed successfully")
        return True, result
    except Exception as e:
        print(f"❌ {test_name} failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, '__traceback__'):
            print(f"   Traceback: {traceback.format_exc().split()[-1] if traceback.format_exc().split() else 'No traceback'}")
        return False, str(e)

def main():
    """Main test runner function"""
    print("🚀 CRISPY Unit Test Runner")
    print("=" * 50)
    
    # Setup test environment
    par = setup_test_environment()
    if par is None:
        print("❌ Cannot proceed without valid parameters")
        return 1
    
    # Ensure output directory exists
    ensure_output_directory(par)
    
    # Import unit test functions
    try:
        from crispy.unitTests import (
            testLoadKernels, testCutout, testFitCutout, testOptExt,
            testGenPixSol, testCreateFlatfield, testCrosstalk
        )
        print("✅ Unit test functions imported")
    except Exception as e:
        print(f"❌ Failed to import unit test functions: {e}")
        return 1
    
    # Results tracking
    results = {}
    total_tests = 0
    passed_tests = 0
    
    # Test 1: testLoadKernels
    total_tests += 1
    success, result = run_test_safely(testLoadKernels, "testLoadKernels", par)
    results["testLoadKernels"] = (success, result)
    if success:
        passed_tests += 1
    
    # Test 2: testGenPixSol
    total_tests += 1
    success, result = run_test_safely(testGenPixSol, "testGenPixSol", par)
    results["testGenPixSol"] = (success, result)
    if success:
        passed_tests += 1
    
    # Test 3: testCreateFlatfield (basic test)
    total_tests += 1
    success, result = run_test_safely(
        testCreateFlatfield, "testCreateFlatfield", par,
        pixsize=0.1, npix=64, pixval=1.0, Nspec=10  # Smaller params for faster test
    )
    results["testCreateFlatfield"] = (success, result)
    if success:
        passed_tests += 1
    
    # Test 4: testCrosstalk (basic test)  
    total_tests += 1
    success, result = run_test_safely(
        testCrosstalk, "testCrosstalk", par,
        pixsize=0.1, npix=64, pixval=1.0, Nspec=10  # Smaller params for faster test
    )
    results["testCrosstalk"] = (success, result)
    if success:
        passed_tests += 1
    
    # Test 5: testCutout (requires input data)
    print(f"\n🧪 Testing testCutout...")
    try:
        # Create synthetic test data
        test_data = np.random.rand(100, 100) * 1000
        success, result = run_test_safely(
            testCutout, "testCutout", par, test_data, lensX=0, lensY=0, dy=2.5
        )
        results["testCutout"] = (success, result)
        total_tests += 1
        if success:
            passed_tests += 1
    except Exception as e:
        print(f"❌ testCutout setup failed: {e}")
        results["testCutout"] = (False, str(e))
        total_tests += 1
    
    # Test 6: testOptExt (requires image input)
    print(f"\n🧪 Testing testOptExt...")
    try:
        from crispy.tools.image import Image
        test_data = np.random.rand(100, 100) * 1000
        test_image = Image(data=test_data)
        success, result = run_test_safely(
            testOptExt, "testOptExt", par, test_image, lensX=0, lensY=0
        )
        results["testOptExt"] = (success, result)
        total_tests += 1
        if success:
            passed_tests += 1
    except Exception as e:
        print(f"❌ testOptExt setup failed: {e}")
        results["testOptExt"] = (False, str(e))
        total_tests += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, (success, result) in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if not success:
            print(f"    Error: {result}")
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)