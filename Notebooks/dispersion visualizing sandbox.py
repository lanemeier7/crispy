# %%
# Section: Setup and test image load
import numpy as np
from crispy.tools.initLogger import getLogger
import glob
from crispy.tools.image import Image

log = getLogger('crispy')
image_filepath = r"C:\Users\ebray\Box\ExoSpec-shared\1_IFS\PISCES\Cal_Data\CRISPY\Calibration\wavecalR70_660_sandbox\PSFloc.fits"
log.info(f"Test log output")
test_image = Image(filename=image_filepath)

# =============================================================
# %% Section: Dispersion visualization from lamsol
# =============================================================
from crispy.tools.wavecal import illustrate_dispersion
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
lamsol_filepath = r"C:\Users\ebray\Box\ExoSpec-shared\1_IFS\PISCES\Cal_Data\CRISPY\Calibration\wavecalR70_660_sandbox\lamsol_sim.dat"
# lamsol_filepath = r"C:\Users\ebray\Github Repos\crispy\crispy\ReferenceFiles\testing\lamsol.dat"

lamsol_df = pd.read_csv(lamsol_filepath, delimiter=' ', engine='python', header=None)
wavelengths, dispersion = illustrate_dispersion([600, 650, 700], lamsol_filepath, nlens=108, output_directory=None)

# =============================================================
# %% Section: Lenslet trace positions vs wavelength
# =============================================================

from crispy.tools.wavecal import transform

order = 3
nlens = 108
x_lens_ind = np.arange(-nlens // 2, nlens // 2)
x_lens_ind, y_lens_ind = np.meshgrid(x_lens_ind, x_lens_ind)

# Display a plot of PSF position vs wavelength for a single lenslet, with the best-fit polynomial overlaid
lenslet_idx = 0  # The middle lenslet index
x_positions = []
y_positions = []
for wavelength in lamsol_df[0]:
    _coefficients_for_transformation = lamsol_df[lamsol_df[0] == wavelength].values[0][1:]
    x_transformed, y_transformed = transform(x_lens_ind, y_lens_ind, order=order, coef=_coefficients_for_transformation)
    x_positions.append(x_transformed[lenslet_idx, lenslet_idx])
    y_positions.append(y_transformed[lenslet_idx, lenslet_idx])

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(lamsol_df[0], x_positions, label='X Position')
ax.plot(lamsol_df[0], y_positions, label='Y Position')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Position (pixels)')
ax.set_title(f'PSF Position vs. Wavelength at Lenslet ({lenslet_idx + nlens // 2}, {lenslet_idx + nlens // 2})')
ax.legend()
fig.tight_layout()

# =============================================================
# %% Section: Test coefficient interpolation utility
# =============================================================

from crispy.tools.wavecal import interpolate_lamsol_wavelengths

# TODO: Revisit crispy.tools.locate_psflets.PSFLets.return_locations() here.
# It may already cover some of the later lamsol interpolation/position analysis,
# and is worth scrutinizing before extending the custom dispersion workflow further.

# Test the polynomial interpolation helper and return a lamsol-like DataFrame.
target_wavelengths = [605.0, 632.5, 677.5]
interpolated_lamsol_df = interpolate_lamsol_wavelengths(
    lamsol_df,
    target_wavelengths=target_wavelengths,
    lamsol_fit_order=4,
    plot_mosaic=True
)

print("Interpolated lamsol rows:")
print(interpolated_lamsol_df.head())

# %%
