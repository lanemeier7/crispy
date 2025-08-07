"""
CRISPY Test Configuration and Fixtures 🧪

This module provides pytest fixtures and configuration for the CRISPY test suite.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add crispy to path for imports
crispy_root = Path(__file__).parent.parent
sys.path.insert(0, str(crispy_root))

@pytest.fixture(scope="session")
def crispy_root_dir():
    """Return the path to the CRISPY root directory"""
    return crispy_root

@pytest.fixture(scope="session") 
def test_data_dir():
    """Return the path to test data directory"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def wfirst_params():
    """Load WFIRST parameters for testing"""
    try:
        from crispy.WFIRST import params
        par = params.Params(codeRoot=str(crispy_root / "crispy"))
        
        # Ensure output directories exist
        os.makedirs(par.exportDir, exist_ok=True)
        os.makedirs(par.unitTestsOutputs, exist_ok=True)
        
        return par
    except Exception as e:
        pytest.skip(f"Cannot load WFIRST parameters: {e}")

@pytest.fixture(scope="session")
def pisces_params():
    """Load PISCES parameters for testing"""
    try:
        from crispy.PISCES import params
        par = params.Params(codeRoot=str(crispy_root / "crispy"))
        
        # Ensure output directories exist
        os.makedirs(par.exportDir, exist_ok=True)
        os.makedirs(par.unitTestsOutputs, exist_ok=True)
        
        return par
    except Exception as e:
        pytest.skip(f"Cannot load PISCES parameters: {e}")

@pytest.fixture
def sample_image_data():
    """Generate sample image data for testing"""
    return np.random.rand(64, 64) * 1000

@pytest.fixture
def large_sample_image_data():
    """Generate larger sample image data for testing"""
    return np.random.rand(100, 100) * 1000

@pytest.fixture
def sample_spectrum_data():
    """Generate sample spectrum data for testing"""
    wavelengths = np.linspace(600, 900, 50)  # nm
    flux = np.random.rand(50) * 1000
    return wavelengths, flux

@pytest.fixture
def crispy_image():
    """Create a CRISPY Image object for testing"""
    try:
        from crispy.tools.image import Image
        data = np.random.rand(64, 64) * 1000
        return Image(data=data)
    except ImportError:
        pytest.skip("Cannot import CRISPY Image class")

@pytest.fixture(scope="session")
def reference_files_available(wfirst_params):
    """Check if reference files are available"""
    if not os.path.exists(wfirst_params.wavecalDir):
        pytest.skip("Reference files not available for testing")
    return True

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add marker for tests that require data files
        if any(keyword in item.name.lower() for keyword in ["kernel", "cutout", "wavecal", "flatfield"]):
            item.add_marker(pytest.mark.requires_data)

def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_data: Tests requiring reference data")
    config.addinivalue_line("markers", "working: Tests known to work")
    config.addinivalue_line("markers", "experimental: Experimental tests")

# Test result emoji mapping
def pytest_runtest_logreport(report):
    """Add emoji to test results"""
    if hasattr(report, 'outcome'):
        if report.outcome == 'passed':
            print("✅", end="")
        elif report.outcome == 'failed':
            print("❌", end="")
        elif report.outcome == 'skipped':
            print("⏭️", end="")

# Session start/end hooks for better output
def pytest_sessionstart(session):
    """Print session start message"""
    print("\n🚀 Starting CRISPY Test Suite")
    print("=" * 50)

def pytest_sessionfinish(session, exitstatus):
    """Print session end message with emoji summary"""
    print("\n" + "=" * 50)
    print("📊 Test Session Complete!")
    
    if exitstatus == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed or had issues")
    
    print("=" * 50)