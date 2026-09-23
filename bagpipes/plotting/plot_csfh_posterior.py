from __future__ import print_function, division, absolute_import
import os 
import numpy as np
import time
import h5py
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *

from .. import utils

from bagpipes import config


try:
    from tqdm import tqdm
except:
    def tqdm(x):
        return x

from scipy.interpolate import RegularGridInterpolator

# Bump when the mah_grid calculation changes so cached grids are recomputed
MAH_GRID_VERSION = 2


def plot_csfh_posterior(fit, show=False, save=True, colorscheme="bw", zvals=[0, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 25]):

    update_rcParams()

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot()

    add_csfh_posterior(fit, ax, colorscheme=colorscheme, zvals=zvals)

    if save:
        print('Saving CSFH plot')
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_csfh.pdf"
        #print(plotpath)
        plt.savefig(plotpath, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax

def add_csfh_posterior_old(fit, ax, colorscheme="bw", z_axis=True, zorder=4, alpha=0.6, plottype='absolute',
                      label=None, zvals=[0, 0.5, 1, 2, 4, 10], color='black', use_color=False, timescale='Gyr', debug = False):

    color1 = "black"
    color2 = "gray"


    if colorscheme == "irnbru":
        color1 = "darkorange"
        color2 = "navajowhite"
        alpha = 0.6

    if colorscheme == "purple":
        color1 = "purple"
        color2 = "purple"
        alpha = 0.4

    if colorscheme == "blue":
        color1 = "dodgerblue"
        color2 = "dodgerblue"
        alpha = 0.7
    
    if use_color:
        color1 = color
        color2 = color
        alpha = alpha 
    # Calculate median redshift and median age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]
  
    if timescale == 'Gyr':
        factor = 10**-9
    elif timescale == 'Myr':
        factor = 10**-6
    elif timescale == 'yr':
        factor = 1
    else:
        raise ValueError("Unit must be Gyr, Myr or yr")
    
    age_of_universe = np.interp(redshift, utils.z_array, utils.age_at_z) *factor/(10**-9)

    times = (fit.posterior.sfh.age_of_universe - fit.posterior.sfh.ages)*10**-9
    print(fit.posterior.sfh.age_of_universe, age_of_universe)
    times_reduced = times[times >= 0]
    
    file = h5py.File(fit.fname[:-1] + ".h5", "a")
    if 'mah_grid' in file.keys():
        mah_grid = file['mah_grid'][:]
        file.close()
    else:

        mah_grid = np.zeros((fit.n_posterior, times.shape[0]))
        grid = RegularGridInterpolator((config.metallicities, config.age_sampling), fit.posterior.sfh.live_frac_grid, fill_value=0.7, bounds_error=False)
        if debug:
            print('Age of Universe', fit.posterior.sfh.age_of_universe/1e9)
            print('Times', np.min(times), np.max(times))

            print('Range of grid:')
            print('Metallicity:', np.min(config.metallicities), np.max(config.metallicities))
            print('Ages:', np.min(config.age_sampling), np.max(config.age_sampling))

        mah_grid = optimize_mah_grid(fit, config, grid)
        file.create_dataset('mah_grid', data=mah_grid, compression='gzip', dtype = np.float32)
        file.close()
    
    masses = np.nanpercentile(mah_grid, (2.5, 16, 50, 84, 97.5), axis=0).T


    
    if plottype == 'absolute':
        ax.plot(times, masses[:, 2], color=color1, zorder=zorder, label=label)
        ax.fill_between(times, masses[:, 1], masses[:, 3], color=color2, alpha=alpha, zorder=zorder)
        ax.fill_between(times, masses[:, 0], masses[:, 4], color=color2, alpha=alpha/2, zorder=zorder)
        ax.set_xlim(age_of_universe, 0)

    elif plottype == 'lookback':
        ax.plot(age_of_universe - times_reduced, masses[times >= 0, 2], color=color1, zorder=zorder, label=label)
        ax.fill_between(age_of_universe - times_reduced, masses[times >= 0, 1], masses[times >= 0, 3], color=color2, alpha=alpha, zorder=zorder)
        ax.fill_between(age_of_universe - times_reduced, masses[times >= 0, 0], masses[times >= 0, 4], color=color2, alpha=alpha/2, zorder=zorder)
 
    ax.set_ylabel(r"$\mathrm{Stellar\,Mass\,(\log_{10}\,M_{\odot}/M_{\star})}$", fontsize='medium')

    if plottype == 'absolute':
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Age\\ of\\ Universe \\ ({timescale})}}}}$", fontsize='medium')

    else:
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Lookback\\ Time \\ ({timescale})}}}}$", fontsize='medium')
    ax.set_yscale('log')

    ax.set_ylim(1e-3*np.nanmax(masses[:, 3]), 1.1*np.nanmax(masses[:, 3]))

    if z_axis:
        if plottype == 'absolute':
            ax2 = add_z_axis(ax, zvals=zvals, reverse = False)


    #ax.set_ylim(0, 1.1*np.nanmax(masses[:, 3]))

def add_csfh_posterior(fit, ax, colorscheme="bw", z_axis=True, zorder=4, alpha=0.6, plottype='absolute',
                      label=None, zvals=[0, 0.5, 1, 2, 4, 10], color='black', use_color=False, timescale='Gyr', debug=False):
    """
    Add cosmic star formation history posterior to an axis.
    Improved to handle time axes consistently and properly display stellar mass evolution.
    """
    # Set colors based on scheme
    color1 = "black"
    color2 = "gray"

    if colorscheme == "irnbru":
        color1 = "darkorange"
        color2 = "navajowhite"
        alpha = 0.6
    elif colorscheme == "purple":
        color1 = "purple"
        color2 = "purple"
        alpha = 0.4
    elif colorscheme == "blue":
        color1 = "dodgerblue"
        color2 = "dodgerblue"
        alpha = 0.7
    
    if use_color:
        color1 = color
        color2 = color
        alpha = alpha 
    
    # Calculate median redshift and age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])
    else:
        redshift = fit.fitted_model.model_components["redshift"]
  
    # Set time unit conversion factor
    if timescale == 'Gyr':
        factor = 10**-9
    elif timescale == 'Myr':
        factor = 10**-6
    elif timescale == 'yr':
        factor = 1
    else:
        raise ValueError("Unit must be Gyr, Myr or yr")
    
    # Universe age at the galaxy's redshift
    age_of_universe = np.interp(redshift, utils.z_array, utils.age_at_z) * factor/(10**-9)
    
    # Convert ages to appropriate time units
    times = (fit.posterior.sfh.age_of_universe - fit.posterior.sfh.ages) * factor
    
    if debug:
        print("Age of Universe at z={}: {} Gyr".format(redshift, age_of_universe))
        print("Time range: {} to {} Gyr".format(np.min(times), np.max(times)))
    
    # Filter times to only include those in the past (relative to galaxy's redshift)
    times_reduced = times[times >= 0]
    
    # Load or calculate MAH grid
    file = h5py.File(fit.fname[:-1] + ".h5", "a")
    cached = ('mah_grid' in file.keys()
              and file['mah_grid'].attrs.get('version', 1) >= MAH_GRID_VERSION)
    if cached:
        mah_grid = file['mah_grid'][:]
        file.close()
    else:
        # Grids written before version 2 summed over the wrong populations
        if 'mah_grid' in file.keys():
            del file['mah_grid']

        if debug:
            print('Age of Universe', fit.posterior.sfh.age_of_universe/1e9)
            print('Range of grid:')
            print('Metallicity:', np.min(config.metallicities), np.max(config.metallicities))
            print('Ages:', np.min(config.age_sampling), np.max(config.age_sampling))

        # Use our fixed function
        mah_grid = optimize_mah_grid_numpy(fit, config)

        # Save result for future use
        dset = file.create_dataset('mah_grid', data=mah_grid, compression='gzip', dtype=np.float32)
        dset.attrs['version'] = MAH_GRID_VERSION
        file.close()
    
    # Calculate percentiles for plotting
    masses = np.nanpercentile(mah_grid, (2.5, 16, 50, 84, 97.5), axis=0).T
    
    # Plot results based on chosen time representation
    if plottype == 'absolute':
        # Absolute time (age of universe)
        ax.plot(times, masses[:, 2], color=color1, zorder=zorder, label=label)
        ax.fill_between(times, masses[:, 1], masses[:, 3], color=color2, alpha=alpha, zorder=zorder)
        ax.fill_between(times, masses[:, 0], masses[:, 4], color=color2, alpha=alpha/2, zorder=zorder)
        ax.set_xlim(age_of_universe, 0)  # Observation on the left, Big Bang on the right, as in plot_sfh_posterior
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Age\\ of\\ Universe \\ ({timescale})}}}}$", fontsize='medium')

    elif plottype == 'lookback':
        # Lookback time (time before observation)
        valid_indices = times >= 0
        lookback_times = fit.posterior.sfh.ages[valid_indices] * factor
        ax.plot(lookback_times, masses[valid_indices, 2], color=color1, zorder=zorder, label=label)
        ax.fill_between(lookback_times, masses[valid_indices, 1], masses[valid_indices, 3], 
                       color=color2, alpha=alpha, zorder=zorder)
        ax.fill_between(lookback_times, masses[valid_indices, 0], masses[valid_indices, 4], 
                       color=color2, alpha=alpha/2, zorder=zorder)
        ax.set_xlim(0, age_of_universe)  # Start at 0 (present) and go back in time
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Lookback\\ Time \\ ({timescale})}}}}$", fontsize='medium')
 
    # Set y-axis properties
    ax.set_ylabel(r"$\mathrm{Stellar\,Mass\,(\log_{10}\,M_{\odot}/M_{\star})}$", fontsize='medium')
    ax.set_yscale('log')
    
    # Set reasonable y-limits based on the data
    max_mass = np.nanmax(masses[:, 3])
    min_mass = np.nanmin(masses[:, 1])
    # Ensure minimum is at least 3 orders of magnitude below maximum for log scale visibility
    min_display = max(min_mass, 1e-3 * max_mass)
    ax.set_ylim(min_display, 1.1 * max_mass)

    # Add redshift axis if requested
    if z_axis:
        if plottype == 'absolute':
            ax2 = add_z_axis(ax, zvals=zvals, reverse=False)

def optimize_mah_grid_old(fit, config, grid):
    '''Fastest version of the MAH grid calculation so far'''
    n_posterior = fit.n_posterior
    n_times = len(fit.posterior.sfh.ages)
    mah_grid = np.zeros((n_posterior, n_times))
    
    assert n_posterior == len(fit.posterior.samples["mass_weighted_zmet"]), f"Number of posterior samples does not match the number of mass-weighted metallicities for {fit.galaxy.ID}, {n_posterior} != {len(fit.posterior.samples['mass_weighted_zmet'])}"

    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"], 
                                 np.min(config.metallicities), 
                                 np.max(config.metallicities))
    
    # Pre-compute age widths * sfh for all samples
    sfh_weighted = fit.posterior.samples["sfh"] * fit.posterior.sfh.age_widths
    
    for k in tqdm(range(n_posterior)):
        # Initialize with the full array for the first step
        alive_frac = grid((mass_weighted_zmet[k], fit.posterior.sfh.ages - fit.posterior.sfh.ages[0]))
        mah_grid[k, 0] = np.sum(sfh_weighted[k] * alive_frac)
        
        cumsum = np.cumsum(sfh_weighted[k][::-1])[::-1]
        
        for l in range(1, n_times):
            # Compute only the new value
            new_alive_frac = grid((mass_weighted_zmet[k], np.array([0])))
            
            # Update mah_grid using the cumulative sum
            mah_grid[k, l] = cumsum[l] * new_alive_frac + np.sum(
                sfh_weighted[k, l:] * (alive_frac[:-l] - new_alive_frac)
            )
            
            # Update alive_frac for the next iteration
            alive_frac[:-1] = alive_frac[1:]
            alive_frac[-1] = new_alive_frac

    return mah_grid

def optimize_mah_grid_numpy(fit, config, grid=None):
    '''
    Mass assembly history (surviving stellar mass) for every posterior sample.

    sfh.ages is lookback time from the observation, so the mass in place at
    lookback time ages[i] comes from populations formed earlier (j >= i),
    each of age ages[j] - ages[i]. Returns an (n_posterior, n_ages) array.

    The live fraction is bilinearly interpolated in (metallicity, age), as
    RegularGridInterpolator would, but ages outside config.age_sampling are
    clamped to the edge rather than filled. ``grid`` is ignored and kept only
    for backwards compatibility.
    '''
    n_posterior = fit.n_posterior

    # Ensure consistent sample count
    assert n_posterior == len(fit.posterior.samples["mass_weighted_zmet"]), \
        f"Number of posterior samples does not match the number of mass-weighted metallicities for {fit.galaxy.ID}, " \
        f"{n_posterior} != {len(fit.posterior.samples['mass_weighted_zmet'])}"

    metallicities = np.asarray(config.metallicities)
    live_frac_grid = fit.posterior.sfh.live_frac_grid

    # Constrain metallicities to the valid range in the model
    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"],
                                 metallicities.min(), metallicities.max())

    # Mass formed in each age bin, shape (n_posterior, n_ages)
    sfh_weighted = fit.posterior.samples["sfh"] * fit.posterior.sfh.age_widths

    # pop_ages[i, j] = age at lookback time ages[i] of stars formed at ages[j]
    ages = fit.posterior.sfh.ages
    pop_ages = ages[np.newaxis, :] - ages[:, np.newaxis]
    formed = pop_ages >= 0.
    pop_ages = np.clip(pop_ages, 0., None)

    # Linear interpolation weights between bracketing metallicity nodes
    z_lo = np.clip(np.searchsorted(metallicities, mass_weighted_zmet) - 1,
                   0, len(metallicities) - 2)
    w_hi = ((mass_weighted_zmet - metallicities[z_lo])
            / (metallicities[z_lo + 1] - metallicities[z_lo]))
    w_lo = 1. - w_hi

    mah_grid = np.zeros(sfh_weighted.shape)
    for z in np.unique(np.concatenate([z_lo, z_lo + 1])):
        weight = w_lo * (z_lo == z) + w_hi * (z_lo + 1 == z)
        live = np.interp(pop_ages, config.age_sampling, live_frac_grid[z])
        live *= formed
        # mah[k, i] = sum_j sfh_weighted[k, j] * live[i, j]
        mah_grid += weight[:, np.newaxis] * (sfh_weighted @ live.T)

    return mah_grid

def optimize_mah_grid1(fit, config, grid):
    '''
    Calculate mass assembly history (MAH) grid accounting for stellar mass loss over time.
    This version properly models how stellar populations lose mass as they age.
    '''
    n_posterior = fit.n_posterior
    n_times = len(fit.posterior.sfh.ages)
    mah_grid = np.zeros((n_posterior, n_times))
    
    # Ensure consistent sample count
    assert n_posterior == len(fit.posterior.samples["mass_weighted_zmet"]), \
        f"Number of posterior samples does not match the number of mass-weighted metallicities for {fit.galaxy.ID}, " \
        f"{n_posterior} != {len(fit.posterior.samples['mass_weighted_zmet'])}"

    # Constrain metallicities to the valid range in the model
    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"], 
                                 np.min(config.metallicities), 
                                 np.max(config.metallicities))
    
    # Pre-compute age widths * sfh for all samples (star formation rate * time interval)
    sfh_weighted = fit.posterior.samples["sfh"] * fit.posterior.sfh.age_widths
    
    for k in tqdm(range(n_posterior)):
        # For each time step, calculate the remaining mass from all previous star formation
        for l in range(n_times):
            current_time = fit.posterior.sfh.ages[l]
            # Sum over all star formation events and their remaining mass fraction
            remaining_mass = 0
            for m in range(l+1):
                # Age of this stellar population at the current time step
                population_age = current_time - fit.posterior.sfh.ages[m]
                # Get surviving fraction based on age and metallicity
                if population_age >= 0:
                    surviving_fraction = grid((mass_weighted_zmet[k], population_age))
                    # Add contribution of this population to the total mass
                    remaining_mass += sfh_weighted[k, m] * surviving_fraction
            
            mah_grid[k, l] = remaining_mass
    
    return mah_grid

'''
def optimize_mah_grid(fit, config, grid):
    n_posterior = fit.n_posterior
    n_times = len(fit.posterior.sfh.ages)
    mah_grid = np.zeros((n_posterior, n_times))
    
    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"], 
                                 np.min(config.metallicities), 
                                 np.max(config.metallicities))
    
    # Pre-compute all age differences
    age_diffs = np.diff(fit.posterior.sfh.ages)
    
    for k in tqdm(range(n_posterior)):
        # Initialize with the full array for the first step
        alive_frac = grid((mass_weighted_zmet[k], fit.posterior.sfh.ages - fit.posterior.sfh.ages[0]))
        mah_grid[k, 0] = np.sum(fit.posterior.samples["sfh"][k] * fit.posterior.sfh.age_widths * alive_frac)
        
        for l in range(1, n_times):
            # Shift alive_frac array and compute only the new value
            alive_frac = np.roll(alive_frac, -1)
            alive_frac[-1] = grid((mass_weighted_zmet[k], np.array([0])))
            
            # Update mah_grid using the shifted alive_frac
            mah_grid[k, l] = np.sum(fit.posterior.samples["sfh"][k, l:] * 
                                    fit.posterior.sfh.age_widths[l:] * 
                                    alive_frac[:-l])
    
    return mah_grid
'''