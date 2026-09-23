"""
Compute the cumulative stellar mass history (surviving stellar mass versus
time, a.k.a. mass assembly history) from an existing bagpipes posterior .h5
file, and save it.

Usage:
    python csfh_from_h5.py pipes/posterior/<run>/<ID>.h5 [--out mah.h5]

Run it from the directory that contains ``pipes/`` (or pass the full path;
the script changes into that directory, since bagpipes locates fits by the
relative path pipes/posterior/<run>/<ID>.h5).
"""
import argparse
import os

import h5py
import numpy as np

import bagpipes as pipes
from bagpipes import config
from bagpipes.plotting.plot_csfh_posterior import optimize_mah_grid_numpy


parser = argparse.ArgumentParser()
parser.add_argument("h5_path", help="bagpipes posterior file, .../pipes/posterior/<run>/<ID>.h5")
parser.add_argument("--out", default=None, help="output file (default: <ID>_mah.h5 next to the input)")
args = parser.parse_args()

# Split .../pipes/posterior/<run>/<ID>.h5 into the working dir, run and ID
h5_path = os.path.abspath(args.h5_path)
work_dir, rel = h5_path.split(os.sep + "pipes" + os.sep + "posterior" + os.sep)
run, galaxy_id = os.path.dirname(rel), os.path.basename(rel)[:-len(".h5")]
out_path = os.path.abspath(args.out) if args.out else h5_path[:-len(".h5")] + "_mah.h5"
os.chdir(work_dir)

# Everything needed to rebuild the fit is stored in the file's attributes
with h5py.File(h5_path, "r") as f:
    fit_instructions = f.attrs["fit_instructions"]
    filt_list = list(f.attrs["filt_list"])
fit_instructions = eval(fit_instructions.replace("array", "np.array")
                        .replace("np.np.", "np."))


# The posterior only needs the SFH, but bagpipes wants a galaxy object, so
# give it placeholder photometry with the right filters.
def load_data(ID):
    return np.ones((len(filt_list), 2))


# bagpipes creates pipes/plots/<run> with os.mkdir, which fails for nested
# run names, so make it here
os.makedirs(os.path.join("pipes", "plots", run), exist_ok=True)

galaxy = pipes.galaxy(galaxy_id, load_data, filt_list=filt_list,
                      spectrum_exists=False)
fit = pipes.fit(galaxy, fit_instructions, run=run)

# Posterior samples are drawn when the fit loads; fits with fewer than 500
# samples use them all, so keep n_posterior in step with that.
fit.n_posterior = fit.posterior.n_samples

# Surviving stellar mass for each posterior sample, shape (n_samples, n_ages)
mah_grid = optimize_mah_grid_numpy(fit, config)

sfh = fit.posterior.sfh
lookback_gyr = sfh.ages * 1e-9
cosmic_time_gyr = (sfh.age_of_universe - sfh.ages) * 1e-9
percentiles = np.nanpercentile(mah_grid, (2.5, 16, 50, 84, 97.5), axis=0)

with h5py.File(out_path, "w") as f:
    f["mah_grid"] = mah_grid.astype(np.float32)
    f["lookback_time_gyr"] = lookback_gyr
    f["cosmic_time_gyr"] = cosmic_time_gyr
    f["percentiles"] = percentiles
    f["percentiles"].attrs["levels"] = [2.5, 16, 50, 84, 97.5]
    # Row k of mah_grid is posterior draw samples2d[posterior_indices[k]]
    f["posterior_indices"] = fit.posterior.indices
    f.attrs["units"] = "mah_grid in Msun; times in Gyr"
    f.attrs["source"] = h5_path

# Sanity check: at the observed epoch this should equal bagpipes' stellar_mass
print(f"log10 M* at observation: {np.log10(percentiles[2, 0]):.3f} "
      f"(bagpipes median stellar_mass {np.median(fit.posterior.samples['stellar_mass']):.3f})")
print(f"Saved {mah_grid.shape} mass assembly grid to {out_path}")

# To plot it instead, the one-liner is:
#     fig, ax = pipes.plot_csfh_posterior(fit, save=False, show=True)
# (this also caches mah_grid inside the posterior .h5 itself)
