from __future__ import print_function, division, absolute_import

import numpy as np
import warnings


from copy import deepcopy
from numpy.polynomial.chebyshev import chebval, chebfit
#added by austind 14/11/23
from scipy.optimize import curve_fit
import astropy.units as u
import astropy.constants as const
import os
from .. import utils

try:
    use_bpass = bool(int(os.environ['use_bpass']))
except KeyError:
    use_bpass = False

if use_bpass:
    print('Setup to use BPASS')
    from .. import config_bpass as config
else:
    from .. import config

from .. import filters
from .. import plotting

from .stellar_model import stellar
from .dust_emission_model import dust_emission
from .dust_attenuation_model import dust_attenuation
from .nebular_model import nebular
from .igm_model import igm
from .agn_model import agn
from .star_formation_history import star_formation_history
from ..input.spectral_indices import measure_index
import importlib


# The Voigt-Hjerting profile based on the numerical approximation by Garcia
def H(a,x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5*x**(-2)
    return H0 - a / np.sqrt(np.pi) /\
    P * ( H0 ** 2 * (4. * P**2 + 7. * P + 4. + Q) - Q - 1.0 )


def addAbs(wl_mod, t, zabs):
    """
    A function that calculates the absorption from foreground source:
        wl_mod: list, wavelength values in units of Å
        t: float, hydrogen column density in units of cm^{-2}
        zabs: float, redshift of absorption source
    Returns:
        exp(-tau): float, absorption fraction
    """
    # Constants
    m_e = 9.1095e-28
    e = 4.8032e-10
    c = 2.998e10
    lamb = 1215.67
    f = 0.416
    gamma = 6.265e8
    broad = 1

    C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
    a = lamb * 1.E-8 * gamma / (4.*np.pi * broad)
    dl_D = broad/c * lamb
    x = (wl_mod/(zabs+1.0) - lamb)/dl_D+0.01

    # Optical depth
    tau = np.array([C_a * t * H(a,x)], dtype=np.float64)
    return np.exp(-tau)[0]


class model_galaxy(object):
    """ Builds model galaxy spectra and calculates predictions for
    spectroscopic and photometric observables.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the model you wish to
        generate.

    filt_list : list - optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only required if photometric output
        is desired.

    spec_wavs : array - optional
        An array of wavelengths at which spectral fluxes should be
        returned. Only required of spectroscopic output is desired.

    spec_units : str - optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.

    phot_units : str - optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.

    extra_model_components - boolean - optional
        whether to calculate non critical outputs -UVJ, beta_C94, M_UV, L_UV_dustcorr, Halpha_EWrest, xi_ion_caseB, ndot_ion_caseB

    index_list : list - optional
        list of dicts containining definitions for spectral indices.
    """

    def __init__(
        self, 
        model_components, 
        filt_list=None, 
        spec_wavs=None,
        spec_units="ergscma",
        phot_units="ergscma",
        index_list=None,
        extra_model_components=False, 
        lines_to_save = ['Halpha', 'Hbeta', 'Hgamma', 'OIII_5007', 'OIII_4959', 'NII_6548', 'NII_6584'],
        line_ratios_to_save = ["OIII_4959+OIII_5007__Hbeta", "Halpha__Hbeta", "Hbeta__Hgamma", "NII_6548+NII_6584__Halpha"],
    ):

        if (spec_wavs is not None) and (index_list is not None):
            raise ValueError("Cannot specify both spec_wavs and index_list.")

        try:
            use_bpass = bool(int(os.environ['use_bpass']))
        except KeyError:
            use_bpass = False

        if use_bpass:
            print('Setup to use BPASS')
            from .. import config_bpass as config
        else:
            from .. import config as config

        importlib.reload(config)
        
        if model_components["redshift"] > config.max_redshift:
            raise ValueError("Bagpipes attempted to create a model with too "
                             "high redshift. Please increase max_redshift in "
                             "bagpipes/config.py before making this model.")

        self.spec_wavs = spec_wavs
        self.filt_list = filt_list
        self.spec_units = spec_units
        self.phot_units = phot_units
        self.index_list = index_list

        if self.index_list is not None:
            self.spec_wavs = self._get_index_spec_wavs(model_components)

        # Create a filter_set object to manage the filter curves.
        if filt_list is not None:
            self.filter_set = filters.filter_set(filt_list)

        # Calculate the optimal wavelength sampling for the model.
        self.wavelengths = self._get_wavelength_sampling()

        # Resample the filter curves onto wavelengths.
        if filt_list is not None:
            self.filter_set.resample_filter_curves(self.wavelengths)

        # Set up a filter_set for calculating rest-frame UVJ magnitudes.
        uvj_filt_list = np.loadtxt(utils.install_dir
                                   + "/filters/UVJ.filt_list", dtype="str")

        self.uvj_filter_set = filters.filter_set(uvj_filt_list)
        self.uvj_filter_set.resample_filter_curves(self.wavelengths)

        # Create relevant physical models.
        self.sfh = star_formation_history(model_components)
        self.stellar = stellar(self.wavelengths)
        self.igm = igm(self.wavelengths)
        self.nebular = False
        self.dust_atten = False
        self.dust_emission = False
        self.agn = False

        self.lines_to_save = lines_to_save
        self.line_ratios_to_save = line_ratios_to_save

        if "nebular" in list(model_components):
            if "velshift" not in model_components["nebular"]:
                model_components["nebular"]["velshift"] = 0.

            self.nebular = nebular(self.wavelengths,
                                   model_components["nebular"]["velshift"])

            if "metallicity" in list(model_components["nebular"]):
                self.neb_sfh = star_formation_history(model_components)

        if "dust" in list(model_components):
            self.dust_emission = dust_emission(self.wavelengths)
            self.dust_atten = dust_attenuation(self.wavelengths,
                                               model_components["dust"])

        if "agn" in list(model_components):
            self.agn = agn(self.wavelengths)

        self.update(model_components, extra_model_components = extra_model_components)

    def _get_wavelength_sampling(self):
        """ Calculate the optimal wavelength sampling for the model
        given the required resolution values specified in the config
        file. The way this is done is key to the speed of the code. """

        max_z = config.max_redshift

        if self.spec_wavs is None:
            self.max_wavs = [(self.filter_set.min_phot_wav
                              / (1.+max_z)),
                             1.01*self.filter_set.max_phot_wav, 10**8]

            self.R = [config.R_other, config.R_phot, config.R_other]

        elif self.filt_list is None:
            self.max_wavs = [self.spec_wavs[0]/(1.+max_z),
                             self.spec_wavs[-1], 10**8]

            self.R = [config.R_other, config.R_spec, config.R_other]

        else:
            if (self.spec_wavs[0] > self.filter_set.min_phot_wav
                    and self.spec_wavs[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1],
                                 self.filter_set.max_phot_wav, 10**8]

                self.R = [config.R_other, config.R_phot, config.R_spec,
                          config.R_phot, config.R_other]

            elif (self.spec_wavs[0] < self.filter_set.min_phot_wav
                  and self.spec_wavs[-1] < self.filter_set.max_phot_wav):

                self.max_wavs = [self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1],
                                 self.filter_set.max_phot_wav, 10**8]

                self.R = [config.R_other, config.R_spec,
                          config.R_phot, config.R_other]

            elif (self.spec_wavs[0] > self.filter_set.min_phot_wav
                  and self.spec_wavs[-1] > self.filter_set.max_phot_wav):

                self.max_wavs = [self.filter_set.min_phot_wav/(1.+max_z),
                                 self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1], 10**8]

                self.R = [config.R_other, config.R_phot,
                          config.R_spec, config.R_other]

            elif (self.spec_wavs[0] < self.filter_set.min_phot_wav
                  and self.spec_wavs[-1] > self.filter_set.max_phot_wav):

                self.max_wavs = [self.spec_wavs[0]/(1.+max_z),
                                 self.spec_wavs[-1], 10**8]

                self.R = [config.R_other, config.R_spec, config.R_other]

        # Generate the desired wavelength sampling.
        x = [1.]

        for i in range(len(self.R)):
            if i == len(self.R)-1 or self.R[i] > self.R[i+1]:
                while x[-1] < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

            else:
                while x[-1]*(1.+0.5/self.R[i]) < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

        return np.array(x)

    def _get_R_curve_wav_sampling(self, oversample=4):
        """ Calculate wavelength sampling for the model to be resampled
        onto in order to apply variable spectral broadening. Only used
        if a resolution curve is supplied in model_components. Has to
        be re-run when a model is updated as it depends on redshift.

        Parameters
        ----------

        oversample : float
            Number of spectral samples per full width at half maximum.
        """

        R_curve = np.copy(self.model_comp["R_curve"])

        if "resolution_p0" in list(self.model_comp):
            x = R_curve[:, 0]
            x = 2.*(x - (x[0] + (x[-1] - x[0])/2.))/(x[-1] - x[0])

            coefs = []
            while "resolution_p" + str(len(coefs)) in list(self.model_comp):
                coefs.append(self.model_comp["resolution_p" + str(len(coefs))])

            poly_coefs = np.array(coefs)

            R_curve[:, 1] *= chebval(x, coefs)

        x = [0.95*self.spec_wavs[0]]

        while x[-1] < 1.05*self.spec_wavs[-1]:
            R_val = np.interp(x[-1], R_curve[:, 0], R_curve[:, 1])
            dwav = x[-1]/R_val/oversample
            x.append(x[-1] + dwav)

        return np.array(x)

    def _get_index_spec_wavs(self, model_components):
        """ Generate an appropriate spec_wavs array for covering the
        spectral indices specified in index_list. """

        min = 9.9*10**99
        max = 0.

        indiv_ind = []
        for j in range(len(self.index_list)):
            if self.index_list[j]["type"] == "composite":
                n = 1
                while "component" + str(n) in list(self.index_list[j]):
                    indiv_ind.append(self.index_list[j]["component" + str(n)])
                    n += 1

            else:
                indiv_ind.append(self.index_list[j])

        for i in range(len(indiv_ind)):
            wavs = np.array(indiv_ind[i]["continuum"]).flatten()

            if "feature" in list(indiv_ind[i]):
                extra_wavs = np.array(indiv_ind[i]["feature"])
                wavs = np.concatenate((wavs, extra_wavs))

            min = np.min([min, np.min(wavs)])
            max = np.max([max, np.max(wavs)])

        min = np.round(0.95*min, 2)
        max = np.round(1.05*max, 2)
        sampling = np.round(np.mean([min, max])/5000., 2)

        return np.arange(min, max*(1. + config.max_redshift), sampling)

    def update(self, model_components, extra_model_components=False):
        """ Update the model outputs to reflect new parameter values in
        the model_components dictionary. Note that only the changing of
        numerical values is supported. 
        Parameters
        ----------
        model_components : dict
            A dictionary containing information about the model you wish to
            generate.
        extra_model_components : boolean - whether to calculate non critical outputs -UVJ, beta_C94, M_UV, L_UV_dustcorr, Halpha_EWrest, xi_ion_caseB
        """


        self.model_comp = model_components
        self.sfh.update(model_components)
        if self.dust_atten:
            self.dust_atten.update(model_components["dust"])

        # If the SFH is unphysical do not caclulate the full spectrum
        if self.sfh.unphysical:
            warnings.warn("The requested model includes stars which formed "
                          "before the Big Bang, no spectrum generated.",
                          RuntimeWarning)

            self.spectrum_full = np.zeros_like(self.wavelengths)
            self.uvj = np.zeros(3)

        else:
            self._calculate_full_spectrum(model_components)

        if self.spec_wavs is not None:
            self._calculate_spectrum(model_components)

        # Add any AGN component:
        if self.agn:
            self.agn.update(self.model_comp["agn"])
            agn_spec = self.agn.spectrum
            agn_spec *= self.igm.trans(self.model_comp["redshift"])

            self.spectrum_full += agn_spec/(1. + self.model_comp["redshift"])

            if self.spec_wavs is not None:
                zplus1 = (self.model_comp["redshift"] + 1.)
                agn_interp = np.interp(self.spec_wavs, self.wavelengths*zplus1,
                                       agn_spec/zplus1, left=0, right=0)

                self.spectrum[:, 1] += agn_interp

        if self.filt_list is not None:
            self._calculate_photometry(model_components["redshift"])

        if not self.sfh.unphysical:
            if extra_model_components:
                self._calculate_uvj_mags()
                self._calculate_beta_C94(model_components)
                self._calculate_M_UV(model_components)
                for frame in ["rest", "obs"]:
                    self._calculate_xi_ion_caseB(model_components, frame = frame)
                    self._calculate_ndot_ion_caseB(model_components, frame = frame)
                    self._save_emission_line_fluxes(model_components, lines = self.lines_to_save, frame = frame)
                    self._save_emission_line_EWs(model_components, lines = self.lines_to_save, frame = frame)
                self._save_line_ratios(model_components, line_ratios = self.line_ratios_to_save)

        # Deal with any spectral index calculations.
        if self.index_list is not None:
            self.index_names = [ind["name"] for ind in self.index_list]

            self.indices = np.zeros(len(self.index_list))
            for i in range(self.indices.shape[0]):
                self.indices[i] = measure_index(self.index_list[i],
                                                self.spectrum,
                                                model_components["redshift"])

    def _calculate_full_spectrum(self, model_comp, add_lines = True):
        """ This method combines the models for the various emission
        and absorption processes to generate the internal full galaxy
        spectrum held within the class. The _calculate_photometry and
        _calculate_spectrum methods generate observables using this
        internal full spectrum. """

        t_bc = 0.01
        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)
        if add_lines:
            em_lines = np.zeros(config.line_wavs.shape)

        if self.nebular:
            grid = np.copy(self.sfh.ceh.grid)

            if "metallicity" in list(model_comp["nebular"]):
                nebular_metallicity = model_comp["nebular"]["metallicity"]
                neb_comp = deepcopy(model_comp)
                for comp in list(neb_comp):
                    if isinstance(neb_comp[comp], dict):
                        neb_comp[comp]["metallicity"] = nebular_metallicity

                self.neb_sfh.update(neb_comp)
                grid = self.neb_sfh.ceh.grid

            # All stellar emission below 912A goes into nebular emission
            spectrum_bc[self.wavelengths < 912.] = 0.
            if add_lines:
                em_lines += self.nebular.line_fluxes(grid, t_bc,
                    model_comp["nebular"]["logU"]) * (1 - model_comp["nebular"].get("fesc", 0))
                self.spectrum_neb = self.nebular.spectrum(grid, t_bc,
                    model_comp["nebular"]["logU"]) * (1 - model_comp["nebular"].get("fesc", 0))
                spectrum_bc += self.spectrum_neb
            else:
                self.spectrum_neb_cont = self.nebular.continuum_spectrum(grid, t_bc,
                    model_comp["nebular"]["logU"]) * (1 - model_comp["nebular"].get("fesc", 0))
                spectrum_bc += self.spectrum_neb_cont

        # Add attenuation due to stellar birth clouds.
        if self.dust_atten:
            dust_flux = 0.  # Total attenuated flux for energy balance.

            # Add extra attenuation to birth clouds.
            eta = 1.
            if "eta" in list(model_comp["dust"]):
                eta = model_comp["dust"]["eta"]
                bc_Av_reduced = (eta - 1.)*model_comp["dust"]["Av"]
                bc_trans_red = 10**(-bc_Av_reduced*self.dust_atten.A_cont/2.5)
                spectrum_bc_dust = spectrum_bc*bc_trans_red
                dust_flux += np.trapz(spectrum_bc - spectrum_bc_dust,
                                      x=self.wavelengths)
                if self.nebular:
                    if add_lines:
                        self.spectrum_neb *= bc_trans_red
                    else:
                        self.spectrum_neb_cont *= bc_trans_red
                spectrum_bc = spectrum_bc_dust
            if self.nebular:
                if add_lines:
                    # Attenuate emission line fluxes.
                    bc_Av = eta*model_comp["dust"]["Av"]
                    em_lines *= 10**(-bc_Av*self.dust_atten.A_line/2.5)
        spectrum += spectrum_bc  # Add birth cloud spectrum to spectrum.
        # Add attenuation due to the diffuse ISM.
        if self.dust_atten:
            trans = 10**(-model_comp["dust"]["Av"]*self.dust_atten.A_cont/2.5)
            dust_spectrum = spectrum*trans
            dust_flux += np.trapz(spectrum - dust_spectrum, x=self.wavelengths)

            spectrum = dust_spectrum
            if add_lines:
                self.spectrum_bc = spectrum_bc*trans
                if self.nebular:
                    self.spectrum_neb *= trans
            else:
                self.spectrum_bc_cont = spectrum_bc*trans
                if self.nebular:
                    self.spectrum_neb_cont *= trans

            # Add dust emission.
            qpah, umin, gamma = 2., 1., 0.01
            if "qpah" in list(model_comp["dust"]):
                qpah = model_comp["dust"]["qpah"]

            if "umin" in list(model_comp["dust"]):
                umin = model_comp["dust"]["umin"]

            if "gamma" in list(model_comp["dust"]):
                gamma = model_comp["dust"]["gamma"]

            spectrum += dust_flux*self.dust_emission.spectrum(qpah, umin,
                                                              gamma)

        spectrum *= self.igm.trans(model_comp["redshift"])

        if "dla" in list(model_comp):
            spectrum *= addAbs(self.wavelengths*self.model_comp["redshift"],
                            self.model_comp["dla"]["t"],
                            self.model_comp["dla"]["zabs"])

        if self.dust_atten:
            if add_lines:
                self.spectrum_bc *= self.igm.trans(model_comp["redshift"])
                if self.nebular:
                    self.spectrum_neb *= self.igm.trans(model_comp["redshift"])
            else:
                self.spectrum_bc_cont *= self.igm.trans(model_comp["redshift"])
                if self.nebular:
                    self.spectrum_neb_cont *= self.igm.trans(model_comp["redshift"])

        # Convert from luminosity to observed flux at redshift z.
        self.lum_flux = 1.
        if model_comp["redshift"] > 0.:
            ldist_cm = 3.086*10**24*np.interp(model_comp["redshift"],
                                              utils.z_array, utils.ldist_at_z,
                                              left=0, right=0)

            self.lum_flux = 4*np.pi*ldist_cm**2

        spectrum /= self.lum_flux*(1. + model_comp["redshift"])

        if self.dust_atten:
            if add_lines:
                if self.nebular:
                    self.spectrum_neb /= self.lum_flux*(1. + model_comp["redshift"])
                self.spectrum_bc /= self.lum_flux*(1. + model_comp["redshift"])
            else:
                if self.nebular:
                    self.spectrum_neb_cont /= self.lum_flux*(1. + model_comp["redshift"])
                self.spectrum_bc_cont /= self.lum_flux*(1. + model_comp["redshift"])
        if add_lines:
            em_lines /= self.lum_flux

        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        spectrum *= 3.826*10**33

        if self.dust_atten:
            if add_lines:
                if self.nebular:
                    self.spectrum_neb *= 3.826*10**33
                self.spectrum_bc *= 3.826*10**33
            else:
                if self.nebular:
                    self.spectrum_neb_cont *= 3.826*10**33
                self.spectrum_bc_cont *= 3.826*10**33
        if add_lines:
            em_lines *= 3.826*10**33
            self.line_fluxes = dict(zip(config.line_names, em_lines))

        if add_lines:
            self.spectrum_full = spectrum
        else:
            self.spectrum_full_cont = spectrum

    def _calculate_full_continuum_spectrum(self, model_comp):
        """ This method combines the models for the various emission
        and absorption processes to generate the internal full galaxy
        continuum spectrum held within the class """
        self._calculate_full_spectrum(model_comp, add_lines = False)

    def _calculate_photometry(self, redshift, uvj=False):
        """ This method generates predictions for observed photometry.
        It resamples filter curves onto observed frame wavelengths and
        integrates over them to calculate photometric fluxes. """

        if self.phot_units == "mujy" or uvj:
            unit_conv = "cgs_to_mujy"

        else:
            unit_conv = None

        if uvj:
            phot = self.uvj_filter_set.get_photometry(self.spectrum_full,
                                                      redshift,
                                                      unit_conv=unit_conv)
        else:
            phot = self.filter_set.get_photometry(self.spectrum_full,
                                                  redshift,
                                                  unit_conv=unit_conv)

        if uvj:
            return phot

        self.photometry = phot

    def _calculate_spectrum(self, model_comp):

        import spectres
        """ This method generates predictions for observed spectroscopy.
        It optionally applies a Gaussian velocity dispersion then
        resamples onto the specified set of observed wavelengths. """

        zplusone = model_comp["redshift"] + 1.

        if "veldisp" in list(model_comp):
            vres = 3*10**5/config.R_spec/2.
            sigma_pix = model_comp["veldisp"]/vres
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
            redshifted_wavs = zplusone*self.wavelengths[k_size:-k_size]

        else:
            spectrum = self.spectrum_full
            redshifted_wavs = zplusone*self.wavelengths

        if "R_curve" in list(model_comp):
            oversample = 4  # Number of samples per FWHM at resolution R
            new_wavs = self._get_R_curve_wav_sampling(oversample=oversample)

            # spectrum = np.interp(new_wavs, redshifted_wavs, spectrum)
            spectrum = spectres.spectres(new_wavs, redshifted_wavs,
                                         spectrum, fill=0)
            redshifted_wavs = new_wavs

            sigma_pix = oversample/2.35  # sigma width of kernel in pixels
            k_size = 4*int(sigma_pix+1)
            x_kernel_pix = np.arange(-k_size, k_size+1)

            kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
            kernel /= np.trapz(kernel)  # Explicitly normalise kernel

            # Disperse non-uniformly sampled spectrum
            spectrum = np.convolve(spectrum, kernel, mode="valid")
            redshifted_wavs = redshifted_wavs[k_size:-k_size]

        # Converted to using spectres in response to issue with interp,
        # see https://github.com/ACCarnall/bagpipes/issues/15
        # fluxes = np.interp(self.spec_wavs, redshifted_wavs,
        #                    spectrum, left=0, right=0)

        fluxes = spectres.spectres(self.spec_wavs, redshifted_wavs,
                                   spectrum, fill=0)

        if self.spec_units == "mujy":
            fluxes /= ((10**-29*2.9979*10**18/self.spec_wavs**2))

        self.spectrum = np.c_[self.spec_wavs, fluxes]

    def _calculate_uvj_mags(self):
        """ Obtain (unnormalised) rest-frame UVJ magnitudes. """
        self.uvj = -2.5*np.log10(self._calculate_photometry(0., uvj=True))
        
    # # added by austind 13/10/23
    # def _calculate_stellar_beta_C94(self, model_comp):
    #     """ This method calculates the UV continuum slope (beta) 
    #     in the 10 Calzetti+1994 filters from the stellar spectrum """
    #     stellar_spectrum = self._calculate_stellar_spectrum(model_comp) # un-normalized stellar spectrum
    #     # perform chi square fitting with curve_fit
    #     self.beta_stellar_C94 = None
    
    def _calculate_beta_C94(self, model_comp):
        """ This method calculates the UV continuum slope (beta) 
        in the 10 Calzetti+1994 filters from the full spectrum """
        # constrain to Calzetti filters
        #print(np.min(self.wavelengths), np.max(self.wavelengths))
        self._calculate_full_continuum_spectrum(model_comp)
        wav_obs_C94, f_lambda_obs_C94 = crop_to_C94_filters(self.wavelengths, self.spectrum_full_cont, model_comp)
        self.beta_C94 = np.array([curve_fit(beta_slope_power_law_func, wav_obs_C94, f_lambda_obs_C94, maxfev = 10_000)[0][1]])

    def _calculate_m_UV(self, model_comp):
        """This method calculates the UV apparent magnitude from the full spectrum in a top-hat filter between 1450<wav_rest<1550 Angstrom. 
        Only gives correctly normalized values when model spectrum has already been fitted."""
        f_lambda_1500 = np.mean(self.spectrum_full[((self.wavelengths > 1_450.) & (self.wavelengths < 1_550.))]) * u.erg / (u.s * (u.cm ** 2) * u.AA)
        self.m_UV = np.array([-2.5 * np.log10((f_lambda_1500 * ((1_500. * (1 + model_comp["redshift"]) * u.AA) ** 2) / const.c).to(u.Jy).value) + 8.9]) # observed frame
    
    def _calculate_M_UV(self, model_comp):
        """This method calculates the UV absolute magnitude from the full spectrum in a top-hat filter between 1450<wav_rest<1550 Angstrom. 
        Only gives correctly normalized values when model spectrum has already been fitted."""
        self._calculate_m_UV(model_comp)
        d_L = utils.cosmo.luminosity_distance(model_comp["redshift"]).to(u.pc)
        self.M_UV = np.array([self.m_UV - 5 * np.log10(d_L.value / 10) \
            + 2.5 * np.log10(1 + model_comp["redshift"])])

    def _calculate_L_UV_dustcorr(self, model_comp, frame = "rest", out_units = u.erg):
        dustcorr_spectrum = self._calculate_full_dustcorr_spectrum(model_comp)
        d_L = utils.cosmo.luminosity_distance(model_comp["redshift"]).to(u.pc)
        # calculate observed frame flux at 1500 Angstrom rest frame
        f_Jy_1500 = (np.mean(dustcorr_spectrum[((self.wavelengths > 1_450.) & \
            (self.wavelengths < 1_550.))]) * u.erg / (u.s * (u.cm ** 2) * u.AA) * \
            ((1_500. * (1. + self.model_comp["redshift"]) * u.AA) ** 2) / const.c).to(u.Jy)
        # calculate L_UV
        L_UV = np.array([4 * np.pi * (f_Jy_1500 * d_L ** 2).to(out_units).value])
        # observed frame == rest frame luminosity (nu)
        setattr(self, f"L_UV_dustcorr_{frame}", L_UV)

    def _calculate_Halpha_EWrest(self, model_comp, line_wav = 6563., delta_wav = 100.):
        # calculate Halpha continuum flux
        dustcorr_cont_spectrum = self._calculate_full_dustcorr_spectrum(model_comp, add_lines = False)
        f_cont_Ha = np.mean(dustcorr_cont_spectrum[((self.wavelengths > line_wav - delta_wav / 2.) & (self.wavelengths < line_wav + delta_wav / 2.))])
        # calculate dust corrected line fluxes and calculate rest frame EW
        self._calculate_dustcorr_em_lines(model_comp)
        self.Halpha_EWrest = np.array([(self.line_fluxes_dustcorr["H  1  6562.81A"] / f_cont_Ha) / (1. + model_comp["redshift"])])

    def _save_emission_line_fluxes(
        self,
        model_comp,
        lines = ['Halpha', 'Hbeta', 'Hgamma', 'OIII_5007', 'OIII_4959', 'NII_6548', 'NII_6584'],
        frame = "rest",
    ):
        if not hasattr(self, f"line_fluxes_dustcorr_{frame}"):
            self._calculate_dustcorr_em_lines(model_comp, frame = frame)
        
        line_names = []
        for line in lines:
            line_dict_name = utils.lines_dict.get(line, False)
            if not line_dict_name:
                if line in getattr(self, f"line_fluxes_dustcorr_{frame}").keys():
                    line_dict_name = line
                else:
                    raise ValueError("The line %s is not in the lines_dict and not in the full dictionary" % line)
            line_flux = np.array([getattr(self, f"line_fluxes_dustcorr_{frame}")[line_dict_name]])
            line_name = f"{line}_flux_{frame}"
            setattr(self, line_name, line_flux)
            line_names.append(line_name)
        self.line_names = line_names

    def _save_emission_line_EWs(
        self,
        model_comp,
        lines = ['Halpha', 'Hbeta', 'Hgamma', 'OIII_5007', 'OIII_4959', 'NII_6548', 'NII_6584'],
        frame = "rest", 
        out_units = u.AA
    ):
        dustcorr_cont_spectrum = self._calculate_full_dustcorr_spectrum(model_comp, add_lines = False)
        self._calculate_dustcorr_em_lines(model_comp, frame = frame)
        
        for line in lines:
            # determine the wavelength for each line in Angstrom
            line_key = utils.lines_dict.get(line, False)
            if line_key:
                line_wav_str = line_key.split(" ")[-1]
                if line_wav_str[-1] == "A":
                    units = u.AA
                elif line_wav_str[-1] == "m":
                    units = u.um
                line_wav = (float(line_wav_str[:-1]) * units).to(u.AA).value
            else:
                raise ValueError("The line %s is not in the lines_dict" % line)

            line_flux = getattr(self, f"line_fluxes_dustcorr_{frame}")[utils.lines_dict[line]]
            line_index = abs(self.wavelengths - line_wav).argmin()
            f_cont_line = dustcorr_cont_spectrum[line_index] # observed frame f_lambda
            
            # save continuum flux in f_nu
            f_cont_line_Jy = (f_cont_line * u.erg / (u.s * u.cm ** 2 * u.AA) \
                * ((self.wavelengths[line_index] * (1. + self.model_comp["redshift"]) \
                * u.AA) ** 2) / const.c).to(u.nJy).value
            setattr(self, f"{line}_cont", np.array([f_cont_line_Jy]))

            if frame == "rest":
                f_cont_line *= (1. + self.model_comp["redshift"]) ** 2
            EW_line = (line_flux * u.AA / f_cont_line).to(out_units).value
            setattr(self, f"{line}_EW_{frame}", np.array([EW_line]))



    def _save_line_ratios(
        self,
        model_comp,
        line_ratios = [
            "OIII_4959+OIII_5007__Hbeta",
            "Halpha__Hbeta",
            "Hbeta__Hgamma",
            "NII_6548+NII_6584__Halpha"
        ],
    ):
        self._calculate_dustcorr_em_lines(model_comp, frame = "rest")
        line_fluxes = getattr(self, "line_fluxes_dustcorr_rest")
        for line_ratio in line_ratios:
            assert line_ratio.count("__") == 1, "Line ratios must contain a single '__'"
            lines_a = line_ratio.split("__")[0].split("+")
            line_fluxes_a = np.sum([line_fluxes[utils.lines_dict[line]] for line in lines_a])
            lines_b = line_ratio.split("__")[1].split("+")
            line_fluxes_b = np.sum([line_fluxes[utils.lines_dict[line]] for line in lines_b])
            setattr(self, line_ratio, np.array([line_fluxes_a / line_fluxes_b]))

    def _calculate_ndot_ion_caseB(self, model_comp, frame = "rest", out_units = u.Hz):
        self._calculate_dustcorr_em_lines(model_comp, frame = frame)
        # calculate luminosity distance
        d_L = utils.cosmo.luminosity_distance(model_comp["redshift"]).to(u.pc)
        # extract Halpha line flux in appropriate frame
        Ha_flux = getattr(self, f"line_fluxes_dustcorr_{frame}")["H  1  6562.81A"] \
            * (u.erg / (u.s * u.cm ** 2))
        # convert line flux to line luminosity
        Ha_lum = 4 * np.pi * Ha_flux * d_L ** 2
        # extract f_esc from model_comp
        f_esc = model_comp["nebular"].get("fesc", 0.)
        # conversion factor for case B Hydrogen recombination
        conv = 7.28e11 / u.erg # (slightly different to 1 / 1.36e-12)
        # calculate ndot_ion
        ndot_ion = np.array([(Ha_lum * conv / (1. - f_esc)).to(out_units).value])
        setattr(self, f"ndot_ion_caseB_{frame}", ndot_ion)

    def _calculate_xi_ion_caseB(self, model_comp, frame = "rest", out_units = u.Hz / u.erg):
        # extract ndot_ion in appropriate frame
        self._calculate_ndot_ion_caseB(model_comp, frame = frame, out_units = u.Hz)
        ndot_ion = getattr(self, f"ndot_ion_caseB_{frame}") * u.Hz
        # extract UV luminosity in appropriate frame
        self._calculate_L_UV_dustcorr(model_comp, frame = frame)
        L_UV = getattr(self, f"L_UV_dustcorr_{frame}") * u.erg
        # calculate xi_ion
        xi_ion = np.array([(ndot_ion / L_UV).to(out_units).value])
        setattr(self, f"xi_ion_caseB_{frame}", xi_ion)
    
    def _calculate_stellar_spectrum(self, model_comp):
        t_bc = 0.01
        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)
        
        # Add attenuation due to stellar birth clouds.
        if self.dust_atten:
            dust_flux = 0.  # Total attenuated flux for energy balance.
        
            # Add extra attenuation to birth clouds.
            eta = 1.
            if "eta" in list(model_comp["dust"]):
                eta = model_comp["dust"]["eta"]
                bc_Av_reduced = (eta - 1.)*model_comp["dust"]["Av"]
                bc_trans_red = 10**(-bc_Av_reduced*self.dust_atten.A_cont/2.5)
                spectrum_bc_dust = spectrum_bc*bc_trans_red
                dust_flux += np.trapz(spectrum_bc - spectrum_bc_dust,
                                      x=self.wavelengths)
        
                spectrum_bc = spectrum_bc_dust
        
        spectrum += spectrum_bc  # Add birth cloud spectrum to spectrum.
        
        # Add attenuation due to the diffuse ISM.
        if self.dust_atten:
            trans = 10**(-model_comp["dust"]["Av"]*self.dust_atten.A_cont/2.5)
            dust_spectrum = spectrum*trans
            dust_flux += np.trapz(spectrum - dust_spectrum, x=self.wavelengths)
            spectrum = dust_spectrum
        
            # Add dust emission.
            qpah, umin, gamma = 2., 1., 0.01
            if "qpah" in list(model_comp["dust"]):
                qpah = model_comp["dust"]["qpah"]
        
            if "umin" in list(model_comp["dust"]):
                umin = model_comp["dust"]["umin"]
        
            if "gamma" in list(model_comp["dust"]):
                gamma = model_comp["dust"]["gamma"]
        
            spectrum += dust_flux*self.dust_emission.spectrum(qpah, umin, gamma)
        
        spectrum *= self.igm.trans(model_comp["redshift"])
        
        # Convert from luminosity to observed flux at redshift z.
        lum_flux = 1.
        if model_comp["redshift"] > 0.:
            ldist_cm = 3.086*10**24*np.interp(model_comp["redshift"],
                                              utils.z_array, utils.ldist_at_z,
                                              left=0, right=0)
            lum_flux = 4*np.pi*ldist_cm**2
        spectrum /= lum_flux*(1. + model_comp["redshift"])
        
        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        self.stellar_spectrum = spectrum * 3.826*10**33
        return self.stellar_spectrum
    
    def _calculate_full_dustcorr_spectrum(self, model_comp, add_lines = True):
        """ This method combines the models for the various emission
        and absorption processes to generate the internal (dust free) 
        full galaxy spectrum held within the class. """

        t_bc = 0.01
        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)

        if self.nebular:
            grid = np.copy(self.sfh.ceh.grid)

            if "metallicity" in list(model_comp["nebular"]):
                nebular_metallicity = model_comp["nebular"]["metallicity"]
                neb_comp = deepcopy(model_comp)
                for comp in list(neb_comp):
                    if isinstance(neb_comp[comp], dict):
                        neb_comp[comp]["metallicity"] = nebular_metallicity

                self.neb_sfh.update(neb_comp)
                grid = self.neb_sfh.ceh.grid

            # All stellar emission below 912A goes into nebular emission
            spectrum_bc[self.wavelengths < 912.] = 0.
            if add_lines:
                spectrum_bc += self.nebular.spectrum(grid, t_bc,
                    model_comp["nebular"]["logU"]) * (1 - model_comp["nebular"].get("fesc", 0))
            else:
                spectrum_bc += self.nebular.continuum_spectrum(grid, t_bc,
                    model_comp["nebular"]["logU"]) * (1 - model_comp["nebular"].get("fesc", 0))

        spectrum += spectrum_bc # Add birth cloud spectrum to spectrum.
        spectrum *= self.igm.trans(model_comp["redshift"])

        # Convert from luminosity to observed flux at redshift z.
        lum_flux = 1.
        if model_comp["redshift"] > 0.:
            ldist_cm = 3.086*10**24*np.interp(model_comp["redshift"],
                                              utils.z_array, utils.ldist_at_z,
                                              left=0, right=0)

            lum_flux = 4*np.pi*ldist_cm**2

        spectrum /= lum_flux * (1. + model_comp["redshift"])

        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        spectrum *= 3.826*10**33

        if add_lines:
            self.spectrum_full_dustcorr = spectrum
        else:
            self.spectrum_full_cont_dustcorr = spectrum
        return spectrum
    
    def _calculate_dustcorr_em_lines(self, model_comp, frame = "rest"):
        """ This method computes dust corrected emission lines """

        t_bc = 0.01
        if "t_bc" in list(model_comp):
            t_bc = model_comp["t_bc"]

        em_lines = np.zeros(config.line_wavs.shape)

        if self.nebular:
            grid = np.copy(self.sfh.ceh.grid)

            if "metallicity" in list(model_comp["nebular"]):
                nebular_metallicity = model_comp["nebular"]["metallicity"]
                neb_comp = deepcopy(model_comp)
                for comp in list(neb_comp):
                    if isinstance(neb_comp[comp], dict):
                        neb_comp[comp]["metallicity"] = nebular_metallicity

                self.neb_sfh.update(neb_comp)
                grid = self.neb_sfh.ceh.grid

            em_lines += self.nebular.line_fluxes(grid, t_bc,
                model_comp["nebular"]["logU"]) * (1 - model_comp["nebular"].get("fesc", 0))

        # Convert from luminosity to observed flux at redshift z.
        lum_flux = 1.
        if model_comp["redshift"] > 0.:
            ldist_cm = 3.086*10**24*np.interp(model_comp["redshift"],
                                              utils.z_array, utils.ldist_at_z,
                                              left=0, right=0)

            lum_flux = 4*np.pi*ldist_cm**2
        # convert to erg/s/cm^2, or erg/s if redshift = 0.
        em_lines *= 3.826*10**33 / lum_flux

        if frame == "rest":
            em_lines *= (1. + model_comp["redshift"])
        
        save_name = f"line_fluxes_dustcorr_{frame}"
        setattr(self, save_name, dict(zip(config.line_names, em_lines)))

    def plot(self, show=True):
        return plotting.plot_model_galaxy(self, show=show)

    def plot_full_spectrum(self, show=True):
        return plotting.plot_full_spectrum(self, show=show)

def beta_slope_power_law_func(wav_rest, A, beta):
    return (10 ** A) * (wav_rest ** beta)

def crop_to_C94_filters(wav_rest, flux_obs, model_comp): # I think this funtion does not require model_comp['redshift'] due to incorrect spectrum scaling
    # Calzetti 1994 filters
    lower_Calzetti_filt = [1268., 1309., 1342., 1407., 1562., 1677., 1760., 1866., 1930., 2400.]
    upper_Calzetti_filt = [1284., 1316., 1371., 1515., 1583., 1740., 1833., 1890., 1950., 2580.]
    Calzetti94_filter_indices = np.logical_or.reduce([(wav_rest > low_lim) & (wav_rest < up_lim) \
                    for low_lim, up_lim in zip(lower_Calzetti_filt, upper_Calzetti_filt)])
    
    wav_obs = wav_rest[Calzetti94_filter_indices] * (1 + model_comp["redshift"])
    flux_obs = flux_obs[Calzetti94_filter_indices]
    return wav_obs, flux_obs
