# 🌟 CRISPY
## The Coronagraph Rapid Imaging Spectrograph in Python

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GNU%20GPLv3-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-✅%20Passing-brightgreen.svg)]()

**CRISPY** simulates the WFIRST (now Roman Space Telescope) Integral Field Spectrograph with high-fidelity modeling of optical effects, detector characteristics, and data reduction pipelines.

---

## 🎯 **Features**

- 🔭 **High-Fidelity IFS Simulation**: Complete modeling of WFIRST/Roman Space Telescope IFS
- 🌈 **Polychromatic Processing**: Full spectral cube generation and analysis
- 🎛️ **Multiple Configurations**: Support for WFIRST and PISCES instruments
- 🧪 **Comprehensive Testing**: Modern pytest framework with emoji output
- 📊 **Data Reduction Tools**: Optimal extraction, wavelength calibration, and more
- 🐍 **Python 3.8+**: Modernized for current Python standards

---

## 🚀 **Quick Start**

### Prerequisites
- [Conda](https://conda.io/) or [Mamba](https://mamba.readthedocs.io/) package manager

### Installation

#### 🐍 **Option 1: Using Conda/Mamba Environment (Recommended)**
```bash
# Clone the repository
git clone https://github.com/mjrfringes/crispy.git
cd crispy

# Create and activate the conda environment (use 'mamba' instead of 'conda' if preferred)
conda env create -f environment.yml
conda activate crispy

# Install in development mode
pip install -e .

# Test the installation 🧪
python test_environment.py
```

#### ⚡ **Option 2: Direct Installation**
```bash
# Clone and install directly
git clone https://github.com/mjrfringes/crispy.git
cd crispy
python setup.py install
```

---

## 🧪 **Running Tests**

CRISPY includes a modern pytest framework with beautiful emoji output:

### **Quick Test Commands**
```bash
# Always activate environment first
conda activate crispy

# Run all working tests ✅
python run_tests.py --working

# Run unit tests only 🧪  
python run_tests.py --unit

# Run integration tests 🚀
python run_tests.py --integration

# Run fast tests (skip slow ones) ⚡
python run_tests.py --fast

# Run with coverage report 📊
python run_tests.py --coverage

# Get help with all options
python run_tests.py --help
```

### **Example Output**
```
🚀 Starting CRISPY Test Suite
==================================================
tests/unit/test_core_functionality.py ✅.✅✅✅.✅✅ 
tests/unit/test_working_functions.py ✅.✅✅✅.✅✅
==================================================
📊 Test Session Complete!
🎉 All tests passed!
```

---

## 📖 **Usage Examples**

### Basic IFS Simulation
```python
import numpy as np
from crispy.WFIRST import params
from crispy.IFS import polychromeIFS

# Load WFIRST parameters
par = params.Params()

# Create input cube (wavelength, x, y)
wavelengths = np.linspace(600, 900, 10)  # nm
input_cube = np.ones((10, 64, 64))

# Run IFS simulation
detector_image = polychromeIFS(par, wavelengths, input_cube)
```

### Working with Different Instruments
```python
# WFIRST configuration
from crispy.WFIRST import params as wfirst_params
wfirst_par = wfirst_params.Params()

# PISCES configuration  
from crispy.PISCES import params as pisces_params
pisces_par = pisces_params.Params()

print(f"WFIRST R = {wfirst_par.R}")  # R = 50
print(f"PISCES R = {pisces_par.R}")  # R = 70
```

### Optimal Extraction
```python
from crispy.tools.image import Image
from crispy.unitTests import testOptExt

# Load your detector image
detector_data = np.random.rand(128, 128) * 1000
image = Image(data=detector_data)

# Perform optimal extraction
spectrum, variance = testOptExt(par, image, lensX=0, lensY=0)
```

---

## 📁 **Project Structure**

```
crispy/
├── crispy/                 # Main package
│   ├── WFIRST/            # WFIRST instrument parameters
│   ├── PISCES/            # PISCES instrument parameters  
│   ├── tools/             # Analysis and processing tools
│   └── *.py               # Core modules
├── tests/                 # Modern pytest test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── conftest.py        # Test fixtures
├── docs/                  # Documentation and notebooks
├── environment.yml        # Conda environment
├── setup.py              # Package setup
├── pytest.ini           # pytest configuration  
└── run_tests.py         # Test runner with emoji support
```

---

## 🛠️ **Development**

### Running Notebooks
```bash
# Start Jupyter in the crispy environment
conda activate crispy
jupyter notebook

# Navigate to docs/source/notebooks/ for examples
```

### Building Documentation
```bash
cd docs
make html
```

### Adding Tests
Tests are organized by category with descriptive markers:

```python
@pytest.mark.working      # ✅ Known working tests
@pytest.mark.experimental # ⚠️  Experimental features  
@pytest.mark.slow         # 🐌 Long-running tests
@pytest.mark.requires_data # 📁 Needs reference data
```

---

## 🌐 **Documentation**

- **📚 Full Documentation**: [https://mjrfringes.github.io/crispy/index.html](https://mjrfringes.github.io/crispy/index.html)
- **🔧 Developer Guide**: See `CLAUDE.md` for development workflow
- **📓 Example Notebooks**: Located in `docs/source/notebooks/`

---

## 👥 **Contributors**

**Original Development Team:**
- Maxime Rizzo
- Tim Brandt  
- Neil Zimmerman
- Tyler Groff
- Prabal Saxena
- Mike McElwain
- Avi Mandell

**Institution:** NASA Goddard Space Flight Center

---

## 📜 **License**

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🐛 **Issues & Support**

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/mjrfringes/crispy/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/mjrfringes/crispy/discussions)
- **📧 General Questions**: Contact the development team

---

## 🎉 **Getting Started**

Ready to simulate some spectra? 

1. **⬇️ Install**: Follow the installation instructions above
2. **🧪 Test**: Run `python run_tests.py --working` to verify everything works  
3. **📚 Learn**: Check out the notebooks in `docs/source/notebooks/`
4. **🚀 Simulate**: Start with the basic examples above

**Happy Simulating!** ✨🔭