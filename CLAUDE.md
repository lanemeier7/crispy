# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation with Mamba (Recommended)
```bash
# Create and activate the crispy conda environment with Python 3.11
mamba env create -f environment.yml
conda activate crispy

# Install the package in development mode
pip install -e .

# Test the installation
python test_environment.py
```

### Installation with Conda (Alternative)
```bash
# Create and activate the crispy conda environment with Python 3.11
conda env create -f environment.yml
conda activate crispy

# Install the package in development mode
pip install -e .
```

### Alternative Installation
```bash
# If you prefer not to use conda
python setup.py install
```

### Documentation
```bash
cd docs
make html
```

### Testing
**Environment Test:**
```bash
# Always activate crispy environment first
conda activate crispy
python test_environment.py
```

**Unit Test Runner (Recommended):**
```bash
conda activate crispy
python run_unit_tests.py  # Runs all unit tests with proper error handling
```

**Individual Unit Tests:**
Tests are located in `crispy/unitTests.py` and use a custom framework rather than pytest:
```bash
conda activate crispy
python -c "
from crispy.WFIRST import params
from crispy.unitTests import testCreateFlatfield, testOptExt
par = params.Params(codeRoot='./crispy')
testCreateFlatfield(par, npix=64, Nspec=10)  # Smaller test
"
```

**Working Tests (as of Python 3.11 update):**
- ✅ testGenPixSol - Generates pixel solutions
- ✅ testCreateFlatfield - Creates polychromatic flatfields  
- ✅ testCrosstalk - Tests crosstalk functionality
- ✅ testOptExt - Tests optimal extraction
- ⚠️ testLoadKernels - Kernel interpolation (some numeric type issues)
- ⚠️ testCutout - PSF cutout functionality (broadcasting issues)

## Core Architecture

CRISPY simulates the WFIRST Integral Field Spectrograph (IFS) with this architecture:

### Main Components
- **crispy/IFS.py**: Core simulation engine with `polychromeIFS()` function
- **crispy/ETC.py**: Exposure Time Calculator
- **crispy/tools/**: Simulation tools organized by functionality:
  - `spectrograph.py`: Spectral dispersion and kernel handling
  - `detector.py`: Detector simulation and readout (`readDetector()`)
  - `lenslet.py`: Lenslet array processing 
  - `wavecal.py`: Wavelength calibration
  - `reduction.py`: Data reduction (`intOptimalExtract()`, `lstsqExtract()`)
  - `image.py`: Image class for scene representation
  - `locate_psflets.py`: PSFLets class for point spread function management

### Instrument Configurations
Each has its own `params.py` file defining simulation parameters:
- **WFIRST/**: Primary WFIRST IFS configuration
- **PISCES/**: PISCES instrument configuration  
- **HCIFS/**: High Contrast IFS variant
- **WFIRST660/**, **WFIRST_tight/**, etc.: Specialized configurations

### Parameter System
All configurations use a `Params` class that defines:
- Wavelength calibration directories (`wavecalDir`)
- Output paths (`exportDir`, `unitTestsOutputs`)  
- Detector settings and optical parameters
- Save flags for different data products

### Key Data Flow
1. Input scenes processed through `Image` class
2. Lenslet array simulation via `processImagePlane()`
3. Spectral dispersion using kernel interpolation
4. Detector simulation with noise models
5. Data reduction and extraction

## Working with Notebooks

Jupyter notebooks in `docs/source/notebooks/` demonstrate usage patterns and provide analysis tools. Key notebooks:
- Introduction.ipynb: Getting started
- IFS_calibration.ipynb: Calibration procedures
- Diagnose.ipynb: Diagnostic tools

## Development Notes

- Uses astropy for FITS handling with pyfits fallback
- Custom logging via `crispy.tools.initLogger`
- Multiprocessing support in `par_utils.py`
- Extensive calibration data in `ReferenceFiles/`
- Python 2/3 compatibility handling for string types