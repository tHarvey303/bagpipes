from __future__ import print_function,  division,  absolute_import

import os
import numpy as np

from astropy.io import fits

from ..utils import *
from ..making import igm_inoue2014

""" Bagpipes configuration for a stitched Maraston stellar population grid,
Kroupa IMF (0.1-100 Msun), photoionised with Cloudy c23.01 where young:

- ages <= 100 Myr: Maraston (2024) Tenc40, incl. Cloudy nebular emission;
- ages  > 100 Myr: Maraston (2013), interpolated in log Z onto the M24
  metallicities and resampled onto the M24 wavelength grid; no nebular
  emission (nebular emission is only used below 30 Myr anyway).

Built from maraston24-Tenc40_kroupa-0.1,100_cloudy-c23.01-sps.hdf5 and
maraston13_kroupa-0.1,100.hdf5 into the Synthesizer grid
maraston24-Tenc40-m13stitch_kroupa-0.1,100_cloudy-c23.01-sps.hdf5, then
converted with synthesizer.grid.Grid.convert_to_bagpipes(logU=-2.0). The
stitching rules are stored in that HDF5 file's attributes. Grid files should
be placed in the bagpipes/models/grids/ directory.

Usage: bagpipes.set_config("maraston24_Tenc40_m13stitch_kroupa_cloudy")

Caveats:
- Ages span 1 Myr - 15 Gyr, but M13 jumps straight from 100 to 200 Myr.
- M24's lowest metallicity (Z = 0.0003) is dropped (below M13's range), so
  the lowest metallicity is 0.1 Zsol. M13's super-solar Z = 0.04 is not used.
- M24 is ~0.1 dex brighter in the far-UV than M13 at 100 Myr, so there is a
  ~0.1 dex FUV step at the join; optical/NIR agree to <~0.04 dex.
- Neither parent grid has a stellar fraction, so live_frac = 1 and stellar
  masses are formed masses, not surviving masses.
- Nebular emission is computed at a single logU = -2 only.
- Spectral sampling is R ~ 300 (Cloudy wavelength grid). """

# Base name of the converted Synthesizer grid files.
grid_name = "maraston24-Tenc40-m13stitch_kroupa-0.1,100_cloudy-c23.01-sps"


""" These variables control the wavelength sampling for models. """

# Sets the maximum redshift the code is set up to calculate models for.
max_redshift = 25.

# Sets the R = lambda/dlambda value for spectroscopic outputs.
R_spec = 1000.

# Sets the R = lambda/dlambda value for photometric outputs.
R_phot = 100.

# Sets the R = lambda/dlambda value for other spectral regions.
R_other = 20.


""" These variables control the age sampling for the stellar and nebular
emission models. """

# Sets the default age sampling for stellar models in log10(Gyr).
age_sampling = np.arange(6., np.log10(cosmo.age(0.).value) + 9., 0.1)

# Set up edge positions for age bins for stellar + nebular models.
age_bins = 10**make_bins(age_sampling, make_rhs=True)[0]
age_bins[0] = 0.
age_bins[-1] = 10**9*cosmo.age(0.).value

# Set up widths for the age bins for the stellar + nebular models.
age_widths = age_bins[1:] - age_bins[:-1]

# Convert the age sampling from log10(Gyr) to Gyr.
age_sampling = 10**age_sampling

# Timescale for the code to calculate SFRs over in years.
sfr_timescale = 10**8  # This is 100 Myr by default


""" Stellar emission models. """

try:
    # Name of the fits file storing the stellar models
    stellar_file = grid_name + "_stellar_grids.fits"

    # The metallicities of the stellar grids in units of Z_Solar (= 0.02),
    # i.e. Z = 0.002, 0.006, 0.014, 0.02.
    metallicities = np.array([0.1, 0.3, 0.7, 1.0])

    stellar_hdul = fits.open(grid_dir + "/" + stellar_file)

    # The wavelengths of the grid points in Angstroms
    wavelengths = stellar_hdul[-1].data.astype(float)

    # The ages of the grid points in years
    raw_stellar_ages = stellar_hdul[-2].data.astype(float)

    # The fraction of stellar mass still living. Axis 0 runs over age,
    # axis 1 over metallicity. All ones for this grid.
    live_frac = stellar_hdul[-3].data

    # The raw stellar grids, one HDU per metallicity, each (age, wavelength).
    raw_stellar_grid = stellar_hdul[1:len(metallicities) + 1]

    # Set up edge positions for metallicity bins for stellar models.
    metallicity_bins = make_bins(metallicities, make_rhs=True)[0]
    metallicity_bins[0] = 0.

except IOError:
    print("Failed to load stellar grids, these should be placed in"
          + " the bagpipes/models/grids/ directory.")


""" Nebular emission models. """

try:
    # Names of files containing the nebular grids.
    neb_cont_file = grid_name + "_nebular_cont_grids.fits"
    neb_line_file = grid_name + "_nebular_line_grids.fits"

    # Names for the emission features to be tracked.
    line_names = np.genfromtxt(grid_dir + "/" + grid_name
                               + "_cloudy_lines.txt",
                               dtype="str", delimiter="\n")

    # Wavelengths of these emission features in Angstroms.
    line_wavs = np.loadtxt(grid_dir + "/" + grid_name
                           + "_cloudy_linewavs.txt")

    # Ages for the nebular emission grids.
    neb_ages = fits.open(grid_dir + "/" + neb_line_file)[1].data[1:, 0]

    # Wavelengths for the nebular continuum grids.
    neb_wavs = fits.open(grid_dir + "/" + neb_cont_file)[1].data[0, 1:]

    # LogU values for the nebular emission grids.
    logU = np.array([-2.])

    # Grid of line fluxes.
    line_grid = [hdu.data for hdu in
                 fits.open(grid_dir + "/" + neb_line_file)]

    # Grid of nebular continuum fluxes.
    cont_grid = [hdu.data for hdu in
                 fits.open(grid_dir + "/" + neb_cont_file)]

except IOError:
    print("Failed to load nebular grids, these should be placed in the"
          + " bagpipes/models/grids/ directory.")


""" Dust emission models (Draine + Li 2007). The *_no_norm grids are
unnormalised; dust_emission_model normalises each template to unit energy. """

try:
    umin_vals = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00,
                          1.20, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00,
                          10.0, 12.0, 15.0, 20.0, 25.0])

    qpah_vals = np.array([0.10, 0.47, 0.75, 1.12, 1.49, 1.77,
                          2.37, 2.50, 3.19, 3.90, 4.58])

    file_umin_only = fits.open(grid_dir + "/dl07_grids_umin_only_no_norm.fits")
    dust_grid_umin_only = [file_umin_only[i].data
                           for i in range(len(qpah_vals) + 1)]

    file_umin_umax = fits.open(grid_dir + "/dl07_grids_umin_umax_no_norm.fits")
    dust_grid_umin_umax = [file_umin_umax[i].data
                           for i in range(len(qpah_vals) + 1)]

except IOError:
    print("Failed to load dust emission grids, these should be placed in the"
          + " bagpipes/models/grids/ directory.")


""" IGM attenuation models, as in config_BC03. """

igm_redshifts = np.arange(0.0, max_redshift + 0.01, 0.01)
igm_wavelengths = np.arange(1.0, 1225.01, 1.0)

if not os.path.exists(grid_dir + "/d_igm_grid_inoue14.fits"):
    igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

else:
    igm_file = fits.open(grid_dir + "/d_igm_grid_inoue14.fits")

    if len(igm_file) != 4:
        igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

    else:
        wav_check = np.min(igm_file[2].data == igm_wavelengths)
        z_check = igm_file[3].data.shape[0] == igm_redshifts.shape[0]

        if not wav_check or not z_check:
            igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

raw_igm_grid = fits.open(grid_dir + "/d_igm_grid_inoue14.fits")[1].data
