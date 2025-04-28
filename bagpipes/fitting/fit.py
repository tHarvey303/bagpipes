from __future__ import print_function, division, absolute_import

import numpy as np
import os
import re
import time
import warnings
import h5py
import contextlib

try:
    use_bpass = bool(int(os.environ['use_bpass']))
except KeyError:
    use_bpass = False

if use_bpass:
    print('Setup to use BPASS')
    from .. import config_bpass as config
else:
    from .. import config

from copy import deepcopy

try:
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        import pymultinest as pmn
    multinest_available = True
except (ImportError, RuntimeError, SystemExit) as e:
    print("Bagpipes: PyMultiNest import failed, fitting will use the Nautilus" +
          " sampler instead.")
    multinest_available = False

try:
    from nautilus import Sampler
    nautilus_available = True
except (ImportError, RuntimeError, SystemExit):
    print("Bagpipes: Nautilus import failed, fitting with Nautilus will be " +
          "unavailable.")
    nautilus_available = False

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0

from .. import utils
from .. import plotting

from .fitted_model import fitted_model
from .posterior import posterior


def _read_multinest_data(filename):
    """
    Read MultiNest data.

    By default, Fortran drops the "E" symbol for 3-digit exponent output
    (e.g., '0.148232-104'). This impacts the output files currently
    being written by MultiNest. For such caes, this reader inserts
    the "E" symbol into the number string so that the number can be
    converted to a float.

    Parameters
    ----------
    filename : str
        The filename to read.
    """
    # count the columns in the first row of data;
    # without a converter, genfromtxt will read 3-digit exponents as np.nan
    ncolumns = np.genfromtxt(filename, max_rows=1).shape[0]

    # insert "E" before the "+" or "-" exponent symbol if it is missing,
    # so the string can be converted to a float
    # '1.148232-104'  -> 1.148232e-104
    # '1.148232+104'  -> 1.148232e+104
    # '-1.148232-104' -> -1.148232e-104
    # '+1.148232+104' -> 1.148232e+104
    # '0.148232-104'  -> 1.482320e-105
    # '0.148232E-10'  -> 1.482320e-011
    # '1.148232'      -> 1.48232e+000
    convert = lambda s: float(re.sub(r'(\d)([\+\-])(\d)', r'\1E\2\3',
                                     s.decode()))
    converters = dict(zip(range(ncolumns), [convert] * ncolumns))

    return np.genfromtxt(filename, converters=converters)



class fit(object):
    """
    Top-level class for fitting models to observational data.

    Interfaces with MultiNest or nautilus to sample from the posterior
    distribution of a fitted_model object. Performs loading and saving of
    results.

    Parameters
    ----------
    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    fit_instructions : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    time_calls : bool - optional
        Whether to print information on the average time taken for
        likelihood calls.

    n_posterior : int - optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    """

    def __init__(self, galaxy, fit_instructions, run=".", time_calls=False,
                 n_posterior=500):

        self.run = run
        self.galaxy = galaxy
        self.fit_instructions = deepcopy(fit_instructions)
        self.n_posterior = n_posterior

        # Set up the directory structure for saving outputs.
        if rank == 0:
            utils.make_dirs(run=run)

        # The base name for output files.
        self.fname = "pipes/posterior/" + run + "/" + self.galaxy.ID + "_"

        # A dictionary containing properties of the model to be saved.
        self.results = {}

        # If a posterior file already exists load it.
        if os.path.exists(self.fname[:-1] + ".h5"):
            file = h5py.File(self.fname[:-1] + ".h5", "r")

            self.posterior = posterior(self.galaxy, run=run,
                                       n_samples=n_posterior)

            fit_info_str = file.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            fit_info_str = fit_info_str.replace("np.np.", "np.")
            self.fit_instructions = eval(fit_info_str)
            try:
                self.config_used = eval(file.attrs["config"])
                if self.config_used['type'] == 'BPASS':
                    os.environ['use_bpass'] = str(int(True))
                elif self.config_used['type'] == 'BC03':
                    os.environ['use_bpass'] = str(int(False))

            except KeyError:
                pass

            for k in file.keys():
                self.results[k] = np.array(file[k])
                if np.sum(self.results[k].shape) == 1:
                    self.results[k] = self.results[k][0]

            if rank == 0:
                print("\nResults loaded from " + self.fname[:-1] + ".h5\n")

        # Set up the model which is to be fitted to the data.
        self.fitted_model = fitted_model(galaxy, self.fit_instructions,
                                         time_calls=time_calls)


    def add_quantities_to_h5(self, get_advanced=False):
        """ Add advanced quantities to the .h5 file. """
        file = h5py.File(self.fname[:-1] + ".h5", "a")

        print('Adding quantites to h5 file.')

        self.results['basic_quantities'] = {i:j for i, j in self.posterior.samples.items() if i in self.posterior.basic_quantity_names}
            # Get quantities
     
        # Attempt to add advanced quantities to the .h5 file
        if get_advanced:
            self.posterior.get_advanced_quantities()

        self.results['advanced_quantities'] = {i:j for i, j in self.posterior.samples.items() if i not in self.posterior.basic_quantity_names}

        for k in self.results.keys():
            if k in ['basic_quantities', 'advanced_quantities']:
                data = self.posterior.samples  
                if k not in file.keys():
                    file.create_group(k)
                for j in self.results[k].keys():
                    if j in file[k].keys():
                        del file[k][j]

                    file[k].create_dataset(j, data=data[j], compression="gzip" if type(data[j]) is np.ndarray else None)

        file.close()

    def fit(self, verbose=False, n_live=400, use_MPI=True,
            sampler="multinest", n_eff=0, discard_exploration=False,
            n_networks=4, pool=1, overwrite_h5=False):
        """ Fit the specified model to the input galaxy data.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.

        sampler : string - optional
            The sampler to use. Available options are "multinest" and
            "nautilus".

        n_eff : float - optional
            Target minimum effective sample size. Only used by nautilus.

        discard_exploration : bool - optional
            Whether to discard the exploration phase to get more accurate
            results. Only used by nautilus.

        n_networks : int - optional
            Number of neural networks. Only used by nautilus.

        pool : int - optional
            Pool size used for parallelization. Only used by nautilus.
            MultiNest is parallelized with MPI.

        """
        if "lnz" in list(self.results) and not overwrite_h5:
            if rank == 0:
                print("Fitting not performed as results have already been"
                      + " loaded from " + self.fname[:-1] + ".h5. To start"
                      + " over delete this file or change run.\n")
            return

        # Figure out which sampling algorithm to use
        sampler = sampler.lower()

        if (sampler == "multinest" and not multinest_available and
                nautilus_available):
            sampler = "nautilus"
            print("MultiNest not available. Switching to nautilus.")

        elif (sampler == "nautilus" and not nautilus_available and
                multinest_available):
            sampler = "multinest"
            print("Nautilus not available. Switching to MultiNest.")

        elif sampler not in ["multinest", "nautilus"]:
            raise ValueError("Sampler {} not supported.".format(sampler))

        elif not (multinest_available or nautilus_available):
            raise RuntimeError("No sampling algorithm could be loaded.")

        if not os.path.exists(self.fname[:-1] + ".h5"):
            # run the fitting if the results are already saved

            if rank == 0 or not use_MPI:
                print("\nBagpipes: fitting object " + self.galaxy.ID + "\n")

                start_time = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"

                if sampler == "multinest":
                    pmn.run(self.fitted_model.lnlike,
                            self.fitted_model.prior.transform,
                            self.fitted_model.ndim, n_live_points=n_live,
                            importance_nested_sampling=False, verbose=verbose,
                            sampling_efficiency="model",
                            outputfiles_basename=self.fname, use_MPI=use_MPI)

                elif sampler == "nautilus":
                    n_sampler = Sampler(self.fitted_model.prior.transform,
                                        self.fitted_model.lnlike, n_live=n_live,
                                        n_networks=n_networks, pool=pool,
                                        n_dim=self.fitted_model.ndim,
                                        filepath=self.fname + ".h5")

                    n_sampler.run(verbose=verbose, n_eff=n_eff,
                                discard_exploration=discard_exploration)

                os.environ["PYTHONWARNINGS"] = ""
        
            if rank == 0 or not use_MPI:
                print(f'Rank 0 for {self.galaxy.ID}', use_MPI)
                runtime = time.time() - start_time

                print("\nCompleted in " + str("%.1f" % runtime) + " seconds.\n")

                # Load MultiNest outputs and save basic quantities to file.
                if sampler == "multinest":
                    multinest_fname = self.fname + 'post_equal_weights.dat'
                    samples2d = _read_multinest_data(multinest_fname)
                    lnz_line = open(self.fname + "stats.dat").readline().split()
                    self.results["samples2d"] = samples2d[:, :-1]
                    self.results["lnlike"] = samples2d[:, -1]
                    self.results["lnz"] = float(lnz_line[-3])
                    self.results["lnz_err"] = float(lnz_line[-1])

                elif sampler == "nautilus":
                    samples2d = np.zeros((0, self.fitted_model.ndim))
                    log_l = np.zeros(0)
                    while len(samples2d) < self.n_posterior:
                        result = n_sampler.posterior(equal_weight=True)
                        samples2d = np.vstack((samples2d, result[0]))
                        log_l = np.concatenate((log_l, result[2]))
                    self.results["samples2d"] = samples2d
                    self.results["lnlike"] = log_l
                    self.results["lnz"] = n_sampler.log_z
                    self.results["lnz_err"] = 1.0 / np.sqrt(n_sampler.n_eff)

                self.results["median"] = np.median(samples2d, axis=0)
                self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                        (16, 84), axis=0)
                
                fit_instructions = str(self.fit_instructions)
                try:
                    use_bpass = bool(int(os.environ['use_bpass']))
                except KeyError:
                    use_bpass = False

                if use_bpass:
                    print('Setup to use BPASS')
                    mtype = 'BPASS'
                    from .. import config_bpass as config
                else:
                    from .. import config
                    mtype = 'BC03'

                config_dict = str({'stellar_file':config.stellar_file, 'neb_cont_file':config.neb_cont_file, 'neb_line_file':config.neb_line_file, 'type':mtype})
                
                os.system("rm " + self.fname + "*")

        else:
            # load results
            file = h5py.File(self.fname[:-1] + ".h5", "r")
            self.results["samples2d"] = np.array(file["samples2d"])
            self.results["lnlike"] = np.array(file["lnlike"])
            self.results["lnz"] = float(np.array(file["lnz"]))
            self.results["lnz_err"] = float(np.array(file["lnz_err"]))
            self.results["median"] = np.array(file["median"])
            self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                        (16, 84), axis=0)
            fit_instructions = file.attrs["fit_instructions"]
            config_dict = file.attrs["config"]
            file.close()
            # move file to 'old' subdir
            old_filename = f"{'/'.join(self.fname.split('/')[:-1])}/old/{self.fname.split('/')[-1][:-1]}.h5"
            #breakpoint()
            os.makedirs('/'.join(old_filename.split('/')[:-1]), exist_ok=True)
            os.system("mv " + self.fname[:-1] + ".h5 " + old_filename)
            # delete old files
            #os.system("rm " + self.fname + "*")

        file = h5py.File(self.fname[:-1] + ".h5", "w")
        # This is necessary for converting large arrays to strings
        np.set_printoptions(threshold=10**7)
        file.attrs["fit_instructions"] = fit_instructions
        file.attrs["config"] = config_dict
        np.set_printoptions(threshold=10**4)

        for k in self.results.keys():
            if k not in ["basic_quantities", "advanced_quantities"]:
                file.create_dataset(k, data=self.results[k], compression="gzip" if type(self.results[k]) is np.ndarray else None)

        file.close()

        # Create a posterior object to hold the results of the fit.
        self.posterior = posterior(self.galaxy, run=self.run,
                                    n_samples=self.n_posterior)
        self.results['basic_quantities'] = {i:j for i, j in self.posterior.samples.items() if i in self.posterior.basic_quantity_names}
        # Get quantities
        try:
            # Attempt to add advanced quantities to the .h5 file
            self.posterior.get_advanced_quantities()
            self.results['advanced_quantities'] = {i:j for i, j in self.posterior.samples.items() if i not in self.posterior.basic_quantity_names}
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

        # Do it again with advanced quantities. 
        file = h5py.File(self.fname[:-1] + ".h5", "w")

        # This is necessary for converting large arrays to strings
        np.set_printoptions(threshold=10**7)
        file.attrs["fit_instructions"] = fit_instructions
        file.attrs["config"] = config_dict
        np.set_printoptions(threshold=10**4)

        for k in self.results.keys():
            if k in ['basic_quantities', 'advanced_quantities']:
                data = self.posterior.samples  
                file.create_group(k)
                for j in self.results[k].keys():
                    if len(data[j]) > 5: # Dont save dummies
                        # if data is array of float64, convert to float32
                        if type(data[j]) == np.ndarray and data[j].dtype == 'float64':
                            sdata = data[j].astype('float32')
                            # if the data goes to infinity, do not convert to float32
                            if np.isinf(sdata).any():
                                sdata = data[j]
                        else:
                            sdata = data[j]
                        file[k].create_dataset(j, data=sdata, compression="gzip" if type(sdata) is np.ndarray else None) 
            else:
                data = self.results[k]
                if k not in file.keys():
                    file.create_dataset(k, data=data, compression="gzip" if type(data) is np.ndarray else None)
        file.close()
        
        self._print_results()

    def _print_results(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """

        print("{:<25}".format("Parameter")
              + "{:>31}".format("Posterior percentiles"))

        print("{:<25}".format(""),
              "{:>10}".format("16th"),
              "{:>10}".format("50th"),
              "{:>10}".format("84th"))

        print("-"*58)

        for i in range(self.fitted_model.ndim):
            print("{:<25}".format(self.fitted_model.params[i]),
                  "{:>10.3f}".format(self.results["conf_int"][0, i]),
                  "{:>10.3f}".format(self.results["median"][i]),
                  "{:>10.3f}".format(self.results["conf_int"][1, i]))

        print("\n")

    def plot_corner(self, show=False, save=True):
        return plotting.plot_corner(self, show=show, save=save)

    def plot_1d_posterior(self, show=False, save=True):
        return plotting.plot_1d_posterior(self, show=show, save=save)

    def plot_sfh_posterior(self, show=False, save=True, colorscheme="bw"):
        return plotting.plot_sfh_posterior(
            self, show=show, save=save,colorscheme=colorscheme)

    def plot_spectrum_posterior(self, show=False, save=True):
        return plotting.plot_spectrum_posterior(self, show=show, save=save)

    def plot_calibration(self, show=False, save=True):
        return plotting.plot_calibration(self, show=show, save=save)
    
    def plot_csfh_posterior(self, show=False, save=True):
        return plotting.plot_csfh_posterior(self, show=show, save=save)
