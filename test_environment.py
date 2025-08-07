#!/usr/bin/env python
"""
Simple test script to verify the CRISPY environment is working properly
"""

import sys
print(f"Python version: {sys.version}")
print("Testing imports...")

# Test basic scientific libraries
try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError as e:
    print("❌ NumPy:", str(e))

try:
    import scipy
    print("✅ SciPy:", scipy.__version__)
except ImportError as e:
    print("❌ SciPy:", str(e))

try:
    import matplotlib
    print("✅ Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("❌ Matplotlib:", str(e))

try:
    import astropy
    print("✅ Astropy:", astropy.__version__)
except ImportError as e:
    print("❌ Astropy:", str(e))

try:
    import photutils
    print("✅ Photutils:", photutils.__version__)
except ImportError as e:
    print("❌ Photutils:", str(e))

# Test CRISPY specific imports
try:
    import crispy
    print("✅ CRISPY package imported")
except ImportError as e:
    print("❌ CRISPY package:", str(e))

try:
    from crispy.tools.image import Image
    print("✅ Image class imported")
except ImportError as e:
    print("❌ Image class:", str(e))

try:
    from crispy.WFIRST import params
    par = params.Params(codeRoot='.')
    print("✅ WFIRST parameters loaded")
except Exception as e:
    print("❌ WFIRST parameters:", str(e))

# Test photutils specific imports
try:
    from photutils.detection import DAOStarFinder
    print("✅ DAOStarFinder imported")
except ImportError as e:
    print("❌ DAOStarFinder:", str(e))

try:
    from photutils.centroids import centroid_com
    print("✅ centroid_com imported")
except ImportError as e:
    print("❌ centroid_com:", str(e))

# Test creating a simple image
try:
    data = np.random.rand(10, 10)
    img = Image(data=data)
    print("✅ Image object created successfully")
except Exception as e:
    print("❌ Image object creation:", str(e))

print("\n🎉 Environment test completed!")