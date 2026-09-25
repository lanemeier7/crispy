# CRISPY

## The Coronagraph and Rapid Imaging Spectrograph in Python

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-GNU%20GPLv3-green.svg)](LICENSE)

**CRISPY** simulates integral field spectrographs - originally built around the since-descoped Roman Space Telescope integral field spectrograph, and since extended to lab testbeds such as PISCES and DST2 - with high-fidelity modeling of optical effects, detector characteristics, and data reduction pipelines. Beyond simulation, CRISPY includes a mature set of tools for wavelength-calibration diagnostics, least-squares spectral extraction, and IFS data-cube visualization built up from real lab and testbed data.

---

## Features

- **IFS Simulation**: modeling of PISCES and DST2 integral field spectrographs, and more.
- **Polychromatic Processing**: full spectral cube generation and analysis
- **Multiple Configurations**: swap between instrument parameter sets (WFIRST, PISCES, DST2, HCIFS) without touching the core pipeline
- **Wavelength Calibration & Troubleshooting**: dispersion-solution fitting (`locate_psflets.py`, `wavecal.py`) with diagnostic plots for evaluating fit quality
- **Data Reduction Tools**: least-squares and optimal extraction, IFS cube visualization, a modern multithreading backend
- **Python 3.8+**

---

## Quick Start

### Prerequisites
- Python 3.8+ and `pip`

### Installation

Install directly into your existing Python environment:

```bash
git clone https://github.com/mjrfringes/crispy.git
cd crispy
pip install -e .
```

A dedicated environment isn't required to use CRISPY for analysis. If you want the full development setup (Jupyter notebooks, Sphinx docs), a conda environment is provided:

```bash
conda env create -f environment.yml
conda activate crispy
pip install -e .
```

---

## Running Tests

CRISPY uses a pytest-based test suite, organized by marker:

```bash
python run_tests.py --working      # known-working tests
python run_tests.py --unit         # unit tests only
python run_tests.py --integration  # integration tests
python run_tests.py --fast         # skip slow tests
python run_tests.py --coverage     # with coverage report
python run_tests.py --help         # all options
```

**TODO:** many tests — particularly those under the `experimental` marker — are not currently passing or complete; the test suite is a work in progress.

---

## Usage Examples

### Basic IFS Simulation
```python
import numpy as np
import os
from crispy.configs.DST2 import params
from crispy.IFS import polychromeIFS
from crispy.tools.image import Image
from astropy.io import fits

# Load DST2 parameters (point to crispy package root)
crispy_root = os.path.dirname(os.path.abspath(__file__))
par = params.Params(codeRoot=crispy_root)

# Create input cube (wavelength, x, y)
wavelengths = np.linspace(600, 900, 10)  # nm
input_cube = np.ones((10, 64, 64))

# Wrap in Image object with required header keywords
header = fits.PrimaryHDU().header
header['PIXSIZE'] = 0.1  # λ/D
header['LAM_C'] = 0.75   # microns (center wavelength)
input_image = Image(data=input_cube, header=header)

# Run IFS simulation
detector_image = polychromeIFS(par, wavelengths, input_image)
print(f"Detector image shape: {detector_image.shape}")
```

### Working with Different Instruments
```python
import os
from crispy.configs.DST2 import params as dst2_params
from crispy.configs.WFIRST import params as wfirst_params

crispy_root = os.path.dirname(os.path.abspath(__file__))

# DST2 configuration
dst2_par = dst2_params.Params(codeRoot=crispy_root)

# WFIRST configuration
wfirst_par = wfirst_params.Params(codeRoot=crispy_root)

print(f"DST2 R = {dst2_par.R}")      # R = 120
print(f"WFIRST R = {wfirst_par.R}")  # R = 50
```

PISCES and HCIFS configurations follow the same pattern (`crispy.configs.PISCES`, `crispy.configs.HCIFS`).

### Optimal Extraction
```python
import numpy as np
import os
from crispy.tools.image import Image
from crispy.configs.DST2 import params as dst2_params
from crispy.unitTests import testOptExt

crispy_root = os.path.dirname(os.path.abspath(__file__))

# Load parameters
par = dst2_params.Params(codeRoot=crispy_root)

# Create mock detector image
detector_data = np.random.rand(128, 128) * 1000
image = Image(data=detector_data)

# Perform optimal extraction
spectrum, variance = testOptExt(par, image, lensX=0, lensY=0)
print(f"Spectrum shape: {spectrum.shape}")
```

### Wavelength Solution Files
The detector mapping stored in `lamsol.dat` is documented in the Sphinx docs as
`Wavelength Solution Files`. That page explains the column layout, coefficient
ordering, lenslet index convention, and the recommended APIs for converting a
given lenslet index and wavelength into detector `(x, y)` coordinates.

---

## Project Structure

```
crispy/
├── crispy/                  # Main package
│   ├── IFS.py                # Core simulation pipeline
│   ├── ETC.py                # Exposure time calculator
│   ├── unitTests.py           # Legacy test utilities
│   ├── configs/               # Instrument parameter sets
│   │   ├── WFIRST/
│   │   ├── PISCES/
│   │   ├── DST2/
│   │   └── HCIFS/
│   ├── tools/                 # Analysis and processing tools
│   │   ├── wavecal.py          # Wavelength calibration
│   │   ├── locate_psflets.py   # Dispersion-solution fitting/diagnostics
│   │   ├── reduction.py        # Spectral extraction (lstsq, optimal)
│   │   ├── postprocessing.py    # Cube visualization and analysis
│   │   └── ...
│   ├── ReferenceFiles/          # Calibration data (PISCES, DST2)
│   └── SimResults/             # Simulation output directory
├── tests/                    # pytest test suite
│   ├── unit/                  # Unit tests (working + experimental)
│   └── integration/            # Integration tests
├── docs/                     # Sphinx documentation and notebooks
├── pyproject.toml            # Package metadata and dependencies
├── environment.yml           # Optional conda dev environment
├── pytest.ini                # pytest configuration
└── run_tests.py               # Test runner
```

---

## Development

### Running Notebooks
```bash
conda activate crispy
jupyter notebook
```
Example notebooks live in `docs/source/notebooks/`.

### Building Documentation
```bash
cd docs
make html
```

### Adding Tests
Tests are organized by category with descriptive markers:

```python
@pytest.mark.working       # known working tests
@pytest.mark.experimental  # experimental / known-broken features
@pytest.mark.slow          # long-running tests
@pytest.mark.requires_data # needs reference data
```

---

## Documentation

- **Full Documentation**: [https://mjrfringes.github.io/crispy/index.html](https://mjrfringes.github.io/crispy/index.html)
- **Wavelength Solution Format**: build the docs locally and open the `Wavelength Solution Files` page for `lamsol.dat` details
- **Developer Guide**: see `CLAUDE.md` for development workflow
- **Example Notebooks**: located in `docs/source/notebooks/`

---

## Contributors

- Maxime Rizzo
- Tim Brandt
- Neil Zimmerman
- Tyler Groff
- Prabal Saxena
- Mike McElwain
- Avi Mandell
- Evan Bray
- Lane Meier

**Institution:** NASA Goddard Space Flight Center

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/mjrfringes/crispy/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/mjrfringes/crispy/discussions)
- **General Questions**: contact the development team

---

## Getting Started

1. **Install**: follow the installation instructions above
2. **Test**: run `python run_tests.py --working` to verify everything works
3. **Learn**: check out the notebooks in `docs/source/notebooks/`
4. **Simulate**: start with the basic examples above
