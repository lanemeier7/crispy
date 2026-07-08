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
