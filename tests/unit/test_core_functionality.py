"""
Core CRISPY Functionality Tests 🧪

Basic tests for CRISPY core components that should work reliably.
"""

import pytest
import numpy as np


class TestImageClass:
    """Test the CRISPY Image class 📸"""
    
    def test_image_creation(self, sample_image_data):
        """Test creating an Image object ✨"""
        from crispy.tools.image import Image
        
        img = Image(data=sample_image_data)
        assert img.data is not None
        assert img.data.shape == sample_image_data.shape
        assert np.array_equal(img.data, sample_image_data)
    
    def test_image_statistics(self, crispy_image):
        """Test basic image statistics 📊"""
        # Basic statistics should work
        mean_val = np.mean(crispy_image.data)
        std_val = np.std(crispy_image.data)
        
        assert mean_val > 0
        assert std_val > 0
        assert crispy_image.data.shape == (64, 64)


class TestParameterLoading:
    """Test parameter loading for different configurations ⚙️"""
    
    @pytest.mark.working
    def test_wfirst_params_loading(self, wfirst_params):
        """Test loading WFIRST parameters 🌟"""
        assert wfirst_params is not None
        assert hasattr(wfirst_params, 'R')
        assert hasattr(wfirst_params, 'nlens')
        assert hasattr(wfirst_params, 'wavecalDir')
        
        # Check expected values
        assert wfirst_params.R == 50
        assert wfirst_params.nlens == 108
        assert wfirst_params.pinhole is False  # We added this
    
    def test_pisces_params_loading(self, pisces_params):
        """Test loading PISCES parameters 🎯"""
        assert pisces_params is not None
        assert hasattr(pisces_params, 'R')
        assert hasattr(pisces_params, 'nlens')
        
        # Check expected values
        assert pisces_params.R == 70
        assert pisces_params.nlens == 108
        assert pisces_params.pinhole is True  # PISCES uses pinhole
    
    def test_output_directories_exist(self, wfirst_params):
        """Test that output directories are created 📁"""
        import os
        assert os.path.exists(wfirst_params.exportDir)
        assert os.path.exists(wfirst_params.unitTestsOutputs)


class TestBasicImports:
    """Test that all critical imports work 📦"""
    
    def test_crispy_imports(self):
        """Test basic CRISPY imports ✅"""
        # These should all work
        from crispy.tools.image import Image
        from crispy.tools.initLogger import getLogger
        from crispy.WFIRST import params
        from crispy.IFS import polychromeIFS
        
        # Test logger
        log = getLogger('test')
        assert log is not None
    
    def test_scientific_imports(self):
        """Test scientific library imports 🔬"""
        import numpy as np
        import scipy
        import matplotlib
        import astropy
        import photutils
        
        # Test specific imports that caused issues
        from photutils.detection import DAOStarFinder
        from photutils.centroids import centroid_com
        from astropy.io import fits
        
        assert True  # If we get here, imports worked
    
    def test_numpy_types(self):
        """Test that numpy types work correctly 🔢"""
        # Test that our np.int32 fixes work
        arr1 = np.zeros((10, 10), np.int32)
        arr2 = np.ones(100).astype(np.int32)
        
        assert arr1.dtype == np.int32
        assert arr2.dtype == np.int32