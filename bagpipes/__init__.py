from __future__ import print_function, division, absolute_import



import os
import importlib
import sys

from .config_utils import set_config

config_name = 'BC03'

try:
    use_bpass = bool(int(os.environ['use_bpass']))
    if use_bpass:
        config_name = '_bpass'
except KeyError:
    try:
        config_name = os.environ['PIPES_CONFIG_NAME']
        if not config_name.startswith('_'):
            config_name = '_' + config_name
    except KeyError:
        pass

print(f"Using configuration: {config_name[1:] if config_name[0]=='_' else config_name}")

config = set_config(config_name, return_config=True, reload=False)

from . import utils

from . import models
from . import fitting
from . import filters
from . import plotting
from . import input
from . import catalogue
from . import moons

from .models.model_galaxy import model_galaxy
from .input.galaxy import galaxy
from .fitting.fit import fit

from .catalogue.fit_catalogue import fit_catalogue
from .catalogue.fit_catalogue_old import fit_catalogue_old


from .plotting import plot_corner, plot_calibration, plot_1d_posterior, plot_spectrum_posterior, plot_sfh_posterior, add_spectrum, plot_sfh, plot_csfh_posterior, plot_galaxy, general, add_sfh_posterior, add_csfh_posterior