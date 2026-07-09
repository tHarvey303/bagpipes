import argparse
import os
from pathlib import Path
from typing import Union
import shutil

import numpy as np
from astropy.io import fits
import unyt

# This import may fail if synthesizer is not installed, but it is required.
from synthesizer import Grid


# Set a limit for ages that produce significant nebular emission
NEBULAR_AGE_LIMIT_YR = 3e7
ZSOL = 0.02  # Solar metallicity in the context of Bagpipes

def _format_np_array(arr: np.ndarray) -> str:
    """Formats a numpy array into a string for writing to a .py file."""
    return f"np.array({np.array2string(arr, separator=', ', threshold=np.inf)})"

def _write_stellar_fits(grid_name: str, output_dir: Path, grid: Grid) -> str:
    """Writes the stellar SPS grid to the correct Bagpipes FITS format."""
    wavelengths_aa = grid.lam.to('Angstrom').value
    ages_yr = grid.axes_values['ages']
    
    if hasattr(grid, 'stellar_fraction'):
        live_frac = grid.stellar_fraction.astype(np.float64)
    else:
        print("No stellar_fraction attribute found in grid; assuming all mass is in living stars.")
        live_frac = np.ones((len(ages_yr), len(grid.axes_values['metallicities'])))

    # Get the spectral data from the synthesizer grid.
    # The shape is (n_ages, n_metallicities, n_wavelengths).
    incident_sed = grid.get_sed_for_full_grid('incident')
    # Could be multi-dimensional if logU

    incident_spectra = incident_sed.llam.to('Lsun/Angstrom').value #/ incident_sed.bolometric_luminosity.to('Lsun')[..., np.newaxis]

    hdul = fits.HDUList([fits.PrimaryHDU()])

    # Add a spectral grid HDU for each metallicity.
    for i, zmet in enumerate(grid.axes_values['metallicities']):
        # Slice the data for the current metallicity: shape (n_ages, n_wavelengths)
        spec_slice = incident_spectra[:, i, :]

        if 'ionisation_parameters' in grid.axes:
            # Shouldn 't make a difference, just take the first logU value
            spec_slice = spec_slice[:, 0, :]  # Assuming logU is the third dimension

        ## Transpose to the required (n_wavelengths, n_ages) shape for Bagpipes.
        spec_data_for_fits = spec_slice.astype('float32')

        hdu = fits.ImageHDU(data=spec_data_for_fits, name=f"ZMET_{zmet/ZSOL:.4f}")
        hdul.append(hdu)

    # Add metadata HDUs in the required reverse order for loading.
    hdul.append(fits.ImageHDU(data=live_frac.astype('float32'), name="LIV_MSTAR_FRAC"))
    hdul.append(fits.ImageHDU(data=ages_yr.astype('float32'), name="STELLAR_AGE_YR"))
    hdul.append(fits.ImageHDU(data=wavelengths_aa.astype('float32'), name="WAVELENGTHS_AA"))

    output_filename = f"{grid_name}_stellar_grids.fits"
    output_path = output_dir / output_filename
    hdul.writeto(output_path, overwrite=True)
    hdul.close()

    print(f"✅ Stellar FITS grid saved to: {output_path}")
    return output_filename


def _write_cloudy_line_files(output_dir: Path, grid: Grid, lines, grid_name: str) -> None:
    """Writes cloudy_lines.txt and cloudy_linewavs.txt."""
    if not grid.has_lines:
        return

    with open(output_dir / f"{grid_name}_cloudy_lines.txt", "w") as f:
        f.write("\n".join(grid.line_ids))

    np.savetxt(output_dir / f"{grid_name}_cloudy_linewavs.txt", lines.lam.to('Angstrom').value, fmt="%.8e")
    print(f"✅ Cloudy line info files saved in: {output_dir}")

def _write_nebular_fits(grid_name: str, output_dir: Path, grid: Grid, logU_values: np.ndarray, lines, incident_spec, neb_cont_spec, line_lums) -> tuple[str, str]:
    """
    Normalizes nebular outputs and writes them to Bagpipes-compatible FITS files.
    This version handles grids with or without an intrinsic logU dimension.
    """
    if not grid.has_lines or neb_cont_spec is None:
        print("Grid lacks nebular data; skipping nebular FITS generation.")
        return "placeholder_lines.fits", "placeholder_cont.fits"

    # --- 1. Get initial data and set up constants ---
    line_wavs = lines.lam
    wav = grid.lam

    age_mask = (grid.axes_values['ages']) < NEBULAR_AGE_LIMIT_YR
    neb_ages_yr = (grid.axes_values['ages'])[age_mask]
    metallicities_zsol = grid.axes_values['metallicities']

    n_neb_ages, n_z, n_logU, n_lines, n_wavs = len(neb_ages_yr), len(metallicities_zsol), len(logU_values), len(line_wavs), len(wav)

    final_line_lums_lsun = np.zeros((n_neb_ages, n_z, n_logU, n_lines))
    final_cont_flux_lsun_a = np.zeros((n_neb_ages, n_z, n_logU, n_wavs))

    # --- 2. Loop and normalize each spectrum individually using unyt ---
    print("Normalizing nebular emission grids (this may take a moment)...")
    for i_age, age_yr in enumerate(neb_ages_yr):
        original_age_idx = np.where(grid.axes_values['ages'] == age_yr)[0][0]
        for i_z, zmet in enumerate(metallicities_zsol):
            for i_logU, logU_val in enumerate(logU_values):
                incident_spec_slice = incident_spec[original_age_idx, i_z, i_logU, :]
                incident_ergs_a = incident_spec_slice.to('erg/s/Angstrom')
                ionizing_mask = wav <= 911.8 * unyt.Angstrom
                input_ionizing_flux = np.trapz(incident_ergs_a[ionizing_mask], x=wav[ionizing_mask])

                current_line_lums = line_lums[original_age_idx, i_z, i_logU, :]
                total_line_output = np.sum(current_line_lums)

                neb_cont_slice = neb_cont_spec[original_age_idx, i_z, i_logU, :]
                neb_cont_ergs_a = neb_cont_slice.to('erg/s/Angstrom')
                total_cont_output = np.trapz(neb_cont_ergs_a, x=wav)

                output_ionizing_flux = total_line_output + total_cont_output
                norm_factor = (input_ionizing_flux / output_ionizing_flux).to('dimensionless').value if output_ionizing_flux > 0 else 0.

                final_line_lums_lsun[i_age, i_z, i_logU, :] = (current_line_lums * norm_factor).to('Lsun').value
                final_cont_flux_lsun_a[i_age, i_z, i_logU, :] = (neb_cont_ergs_a * norm_factor).to('Lsun/Angstrom').value

    # --- 3. Build and save FITS files ---
    hdul_line = fits.HDUList([fits.PrimaryHDU()])
    hdul_cont = fits.HDUList([fits.PrimaryHDU()])
    for i_z, zmet in enumerate(metallicities_zsol):
        for i_logU, logU_val in enumerate(logU_values):
            # Create Line FITS HDU
            hdu_data_line = np.full((n_neb_ages + 1, n_lines + 1), 0, dtype='float32')
            hdu_data_line[0, 1:] = line_wavs.to('Angstrom').value
            hdu_data_line[1:, 0] = neb_ages_yr
            hdu_data_line[1:, 1:] = final_line_lums_lsun[:, i_z, i_logU, :]
            hdu_line = fits.ImageHDU(hdu_data_line, name=f"ZMET_{zmet:.4f}ZSOL_LOGU_{logU_val:.2f}")
            hdul_line.append(hdu_line)

            # Create Continuum FITS HDU
            hdu_data_cont = np.full((n_neb_ages + 1, n_wavs + 1), 0, dtype='float32')
            hdu_data_cont[0, 1:] = wav.to('Angstrom').value
            hdu_data_cont[1:, 0] = neb_ages_yr
            hdu_data_cont[1:, 1:] = final_cont_flux_lsun_a[:, i_z, i_logU, :]
            hdu_cont = fits.ImageHDU(hdu_data_cont, name=f"ZMET_{zmet:.4f}ZSOL_LOGU_{logU_val:.2f}")
            hdul_cont.append(hdu_cont)

    line_filename = f"{grid_name}_nebular_line_grids.fits"
    hdul_line.writeto(output_dir / line_filename, overwrite=True)
    print(f"✅ Nebular line grid saved to: {output_dir / line_filename}")

    cont_filename = f"{grid_name}_nebular_cont_grids.fits"
    hdul_cont.writeto(output_dir / cont_filename, overwrite=True)
    print(f"✅ Nebular continuum grid saved to: {output_dir / cont_filename}")

    return line_filename, cont_filename



def _generate_config_py_content(stellar_file, neb_cont_file, neb_line_file, grid, grid_name, logU_values) -> str:
    """Generates the full content for the Bagpipes config.py file."""
    metallicities_str = _format_np_array(grid.axes_values['metallicities']/ZSOL)  # Convert to solar metallicity units
    ages_yr_str = _format_np_array(grid.axes_values['ages'])
    n_metal_hdus = len(grid.axes_values['metallicities'])
    logu_str = _format_np_array(np.array(logU_values))

    return f"""
# Bagpipes Configuration File
# Automatically generated for grid: {Path(grid.grid_filename).name}

from __future__ import print_function, division, absolute_import
import os
import numpy as np
from astropy.io import fits

try:
    from ..utils import *
    from ..models.making import igm_inoue2014
except ImportError:
    def make_bins(arr, make_rhs=False):
        bins = 0.5 * (arr[1:] + arr[:-1])
        bins = np.insert(bins, 0, arr[0] - (bins[0] - arr[0]))
        if make_rhs: bins = np.append(bins, arr[-1] + (arr[-1] - bins[-1]))
        return bins, arr


# ======================= AGE SAMPLING ==============================
age_sampling = np.arange(6., np.log10(cosmo.age(0.).value) + 9., 0.1)
age_bins = 10**make_bins(age_sampling, make_rhs=True)[0]
age_bins[0] = 0.
age_bins[-1] = 10**9*cosmo.age(0.).value
age_widths = age_bins[1:] - age_bins[:-1]
age_sampling = 10**age_sampling

# ===================== STELLAR EMISSION MODELS =====================
try:
    stellar_file = "{stellar_file}"
    full_stellar_path = os.path.join(grid_dir, stellar_file)
    metallicities = {metallicities_str}
    raw_stellar_ages = {ages_yr_str}
    wavelengths = fits.open(full_stellar_path)[-1].data
    live_frac = fits.open(full_stellar_path)[-3].data
    raw_stellar_grid = fits.open(full_stellar_path)[1:{n_metal_hdus + 1}]
    metallicity_bins = make_bins(metallicities, make_rhs=True)[0]
    metallicity_bins[0] = 0.
except (IOError, FileNotFoundError):
    print(f"Failed to load stellar grids from {{stellar_file}}")

# ===================== NEBULAR EMISSION MODELS =====================
try:
    neb_cont_file = "{neb_cont_file}"
    neb_line_file = "{neb_line_file}"
    line_names = np.genfromtxt(os.path.join(grid_dir, f"{grid_name}_cloudy_lines.txt"), dtype="str", delimiter="\\n")
    line_wavs = np.loadtxt(os.path.join(grid_dir,  f"{grid_name}_cloudy_linewavs.txt"))
    logU = {logu_str}

    line_grid = [hdu.data for hdu in fits.open(os.path.join(grid_dir, neb_line_file))]
    cont_grid = [hdu.data for hdu in fits.open(os.path.join(grid_dir, neb_cont_file))]

    neb_ages = fits.open(os.path.join(grid_dir, neb_line_file))[1].data[1:, 0]
    neb_wavs = fits.open(os.path.join(grid_dir, neb_cont_file))[1].data[0, 1:]

except (IOError, FileNotFoundError):
    print("Failed to load nebular grids. Ensure generated files are present.")
    line_names, line_wavs, logU, line_grid, cont_grid = [], [], [], [], []

# ===================== OTHER CONFIG (Defaults) =====================
max_redshift = 10.
R_spec, R_phot, R_other = 1000., 100., 20.
sfr_timescale = 1e8


# ===================== DUST EMISSION MODELS (Defaults) =====================
try:
    umin_vals = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00,
                          1.20, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00,
                          10.0, 12.0, 15.0, 20.0, 25.0])
    qpah_vals = np.array([0.10, 0.47, 0.75, 1.12, 1.49, 1.77,
                          2.37, 2.50, 3.19, 3.90, 4.58])
    dust_grid_umin_only = [
        fits.open(os.path.join(grid_dir, "dl07_grids_umin_only.fits"))[i].data for i
        in range(len(qpah_vals) + 1)]
    dust_grid_umin_umax = [
        fits.open(os.path.join(grid_dir, "dl07_grids_umin_umax.fits"))[i].data for i
        in range(len(qpah_vals) + 1)]
except (IOError, FileNotFoundError):
    print("Warning: Default Draine & Li (2007) dust grids not found.")
    dust_grid_umin_only, dust_grid_umin_umax = [], []

# ===================== IGM ATTENUATION (Defaults) =====================
try:
    igm_redshifts = np.arange(0.0, max_redshift + 0.01, 0.01)
    igm_wavelengths = np.arange(1.0, 1225.01, 1.0)
    igm_file_path = os.path.join(grid_dir, "d_igm_grid_inoue14.fits")
    if not os.path.exists(igm_file_path):
        print("Default IGM grid not found, attempting to generate it...")
        igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)
    raw_igm_grid = fits.open(igm_file_path)[1].data
except Exception:
    print("Warning: Could not load or generate default IGM attenuation grid.")
    raw_igm_grid = None
"""

def create_bagpipes_config_from_synth(
    grid_name: str,
    output_dir: Union[str, Path] = "bagpipes_output",
    synth_grid_dir: Union[str, Path, None] = None,
    logU: float = -2.0,
) -> None:
    """Generates Bagpipes configuration files from a Synthesizer grid.
    """
    if Grid is None: return

    if synth_grid_dir is None:
        if 'SYNTHESIZER_GRID_DIR' in os.environ:
            synth_grid_dir = os.environ['SYNTHESIZER_GRID_DIR']

    output_path = Path(output_dir)
    grid_output_path = output_path / "grids"
    grid_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Synthesizer grid: {grid_name}")
    try:
        grid = Grid(grid_name, grid_dir=synth_grid_dir)
    except FileNotFoundError:
        print(f"ERROR: Synthesizer grid '{grid_name}' not found.")
        return

    # Get line collection once for use in multiple functions
    lines = grid.get_lines_for_full_grid()

    base_name = grid.grid_name

    line_lums = lines.luminosity.to('Lsun')
    neb_cont_sed = grid.get_sed_for_full_grid('nebular_continuum')
    neb_cont_spec =neb_cont_sed.llam.to('Lsun/Angstrom')#/neb_cont_sed.bolometric_luminosity.to('Lsun')[..., np.newaxis]
    incident_sed = grid.get_sed_for_full_grid('incident')
    incident_spec = incident_sed.llam.to('Lsun/Angstrom')#/incident_sed.bolometric_luminosity.to('Lsun')[..., np.newaxis]

    if 'ionisation_parameters' in grid.axes:
        print("Found 'logU' axis in the synthesizer grid.")
        logU_values = np.log10(grid.axes_values['ionisation_parameters'])
        # Data from grid already has the logU dimension. We just need to align axes.
        # Synthesizer order: (age, z, logU, ...)
    else:
        logU_values = np.array([logU])
        line_lums = line_lums[:, :, np.newaxis, :]
        neb_cont_spec = neb_cont_spec[:, :, np.newaxis, :]
        incident_spec = incident_spec[:, :, np.newaxis, :]


    stellar_file = _write_stellar_fits(base_name, grid_output_path, grid)
    _write_cloudy_line_files(grid_output_path, grid, lines, grid_name)
    line_file, cont_file = _write_nebular_fits(base_name, grid_output_path, grid,
                                            logU_values, lines, incident_spec,
                                            neb_cont_spec, line_lums)

    config_content = _generate_config_py_content(
        stellar_file, cont_file, line_file, grid, grid_name, logU_values
    )

    # Sanitize the base name for the config file
    base_name = base_name.replace(" ", "_").replace("-", "_").replace(".", "_")
    config_py_path = output_path / f"config_{base_name}.py"
    with open(config_py_path, 'w') as f:
        f.write(config_content)

    print("\n🚀 Generation complete!")
    print(f"Bagpipes config file saved to: {config_py_path}")
    print(f"Associated grid files are in: {grid_output_path}")

def move_to_bagpipes(
    output_dir: Union[str, Path] = "bagpipes_output",
    grid_name: str = "synth_grid"
) -> None:
    """Moves the generated files to the Bagpipes directory."""
    try:
        import bagpipes
        bagpipes_dir = Path(bagpipes.__path__[0])
        target_config_dir = Path(f'{bagpipes_dir}/configs/')
        target_grid_dir = bagpipes_dir / "models/grids/"

        san_grid_name = grid_name.replace(" ", "_").replace("-", "_").replace(".", "_")
        print('Sanitized grid name:', san_grid_name)
        config_name = f"config_{san_grid_name.replace('_hdf5', '.py')}"
        config_path = Path(output_dir) / config_name
        shutil.move(config_path, target_config_dir / config_name)

        # move all files in grids which start with grid_name
        grid_files = list(Path(output_dir).glob(f"grids/{grid_name}_*"))
        for grid_file in grid_files:
            shutil.move(grid_file, target_grid_dir / grid_file.name)

        print(f"✅ Moved config file to: {target_config_dir}")
        print(f"✅ Moved grid files to: {target_grid_dir}")
    except ImportError:
        print("ERROR: Bagpipes package not found. Please ensure Bagpipes is installed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a Synthesizer grid to a Bagpipes config and FITS file."
    )
    parser.add_argument("grid_name", type=str, help="Name of the Synthesizer grid file.")
    parser.add_argument(
        "--output_dir", type=str, default="bagpipes_output",
        help="Directory to save the generated files."
    )
    parser.add_argument(
        "--synth_grid_dir", type=str, default=None,
        help="Path to the Synthesizer grid HDF5 file."
    )
    parser.add_argument(
        "--logU", type=float, default=-2.0,
        help="The fixed log(U) value for nebular emission."
    )
    parser.add_argument(
        "--move_to_bagpipes", type=bool, default=False,
        help="If True, moves the generated files to the Bagpipes directory."
    )

    args = parser.parse_args()
    create_bagpipes_config_from_synth(
        grid_name=args.grid_name,
        output_dir=args.output_dir,
        synth_grid_dir=args.synth_grid_dir,
        logU=args.logU
    )

    if args.move_to_bagpipes:
        move_to_bagpipes(output_dir=args.output_dir, grid_name=args.grid_name)
