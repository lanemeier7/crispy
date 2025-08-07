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

### Testing 🧪

**Modern pytest Framework (Recommended):**
```bash
# Always activate crispy environment first
conda activate crispy

# Run all tests with beautiful emoji output
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m working       # Known working tests
pytest -m "not experimental"  # Skip experimental tests

# Run with coverage
pytest --cov=crispy

# Run tests in parallel (faster)
pytest -n auto

# Verbose output with test names
pytest -v

# Run specific test files
pytest tests/unit/test_core_functionality.py
pytest tests/unit/test_working_functions.py
```

**Legacy Test Runners (Still Available):**
```bash
# Environment verification
python test_environment.py

# Original unit test runner  
python run_unit_tests.py
```

**Test Categories:**
- 🧪 **Unit Tests** (`tests/unit/`):
  - `test_core_functionality.py` - Basic imports, parameters, Image class
  - `test_working_functions.py` - Proven working functions (pixel solutions, flatfields, extraction)
  - `test_experimental.py` - Functions with known issues (kernel loading, cutouts)

- 🚀 **Integration Tests** (`tests/integration/`):
  - `test_full_pipeline.py` - End-to-end workflows, multi-configuration tests

**Test Markers:**
- ✅ `@pytest.mark.working` - Reliably working tests
- ⚠️ `@pytest.mark.experimental` - Tests with known issues  
- 🐌 `@pytest.mark.slow` - Long-running tests
- 📁 `@pytest.mark.requires_data` - Tests needing reference files
- 🧪 `@pytest.mark.unit` - Unit tests
- 🚀 `@pytest.mark.integration` - Integration tests

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