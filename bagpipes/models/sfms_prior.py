"""Star-forming main sequence (SFMS) based prior for non-parametric SFHs.

Used by ``star_formation_history.sfms``.  The star-formation history is built in
lookback-time bins, oldest first: in every bin the SFR is the literature SFMS
value for the stellar mass assembled so far and the redshift of the bin centre,
times a log-normal deviation.  Deviations in neighbouring bins are correlated
through a Gaussian-process kernel (Tacchella+2020 regulator model; Wan+2024
extended regulator model) and are parametrised by standard-normal variables
``u1 ... uN`` so that every fitted parameter keeps an independent prior.  The
total formed mass is fixed by ``massformed``: the "seed" mass present at the
start of the oldest bin is solved for so that the track integrates to the
requested mass.  Bins, SFMS values and kernel all follow the sampled redshift.

Everything that does not depend on the galaxy (the SFMS on a (log M*, z) grid,
Cholesky factors of the kernel for a given set of bin centres) is tabulated and
cached.

SFMS relations (all give log10 SFR / Msun yr^-1 for log10 M*/Msun and z):

``speagle14``        Speagle+2014 eq. 28 (Kroupa IMF, t = cosmic age in Gyr).
``popesso23``        Popesso+2023 eq. 14 with turnover (Kroupa, 0 < z < 6).
``koprowski24``      Koprowski+2024 eq. with exponential z evolution (Chabrier, z < 6.75).
``simmonds25_100myr`` / ``simmonds25_10myr``  Simmonds+2025 JADES z = 3-9 sSFR fit
                     (Table 1; SFR averaged over 100 or 10 Myr, log M* 9-10.3).
``clarke24_uv`` / ``clarke24_ha``  Clarke+2024 JADES+CEERS per-z-bin fits
                     (UV ~100 Myr, Halpha ~5 Myr), interpolated in z, clamped outside 1.4 < z < 7.

Chabrier and Kroupa IMFs differ by < 0.03 dex in SFR/M*; no correction is applied.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from .. import utils

LOGM_GRID = np.arange(3.0, 13.5001, 0.05)
Z_GRID = np.arange(0.0, 30.0001, 0.02)

_SFMS_TABLES = {}
_CHOL_CACHE = {}
_CHOL_CACHE_MAX = 20000


# ---------------------------------------------------------------------------
# Cosmic time <-> redshift (Bagpipes cosmology, FlatLambdaCDM H0=70 Om0=0.3)
# ---------------------------------------------------------------------------

def age_at_z(z):
    """Cosmic age in Gyr at redshift z (scalar or array)."""
    return np.interp(z, utils.z_array, utils.age_at_z)


def z_at_age(t_gyr):
    """Redshift at cosmic age t_gyr (scalar or array)."""
    return np.interp(t_gyr, utils.age_at_z[::-1], utils.z_array[::-1])


# ---------------------------------------------------------------------------
# Literature SFMS relations
# ---------------------------------------------------------------------------

def _speagle14(logm, z):
    t = age_at_z(z)
    return (0.84 - 0.026 * t) * logm - (6.51 - 0.11 * t)


def _popesso23(logm, z):
    a0, a1, a2, a3, a4 = 2.693, -0.186, 10.85, -0.0729, 0.99
    t = age_at_z(z)
    return a0 + a1 * t - np.log10(1. + (10 ** (logm - (a2 + a3 * t))) ** (-a4))


def _koprowski24(logm, z):
    a1, a2, a3 = 3.47, -3.13, 0.56
    b1, b2, b3 = 11.69, -1.66, 0.53
    log_sfrmax = a1 + a2 * np.exp(-a3 * z)
    log_m0 = b1 + b2 * np.exp(-b3 * z)
    return log_sfrmax - np.log10(1. + 10 ** (log_m0 - logm))


def _simmonds25(logm, z, sb, beta, mu):
    # sSFR [Gyr^-1] = sb (M/1e10)^beta (1+z)^mu
    return np.log10(sb) - 9. + logm + beta * (logm - 10.) + mu * np.log10(1. + z)


_CLARKE_ZC = np.array([2.05, 3.35, 4.5, 5.5, 6.5])
_CLARKE = {  # slope m, intercept b at log M* = 9.16 (Table 1)
    "uv": (np.array([0.59, 0.68, 0.56, 0.53, 0.64]), np.array([0.32, 0.63, 0.68, 0.88, 0.88])),
    "ha": (np.array([0.95, 0.67, 0.74, 0.75, 0.69]), np.array([0.16, 0.59, 0.81, 0.83, 0.97])),
}


def _clarke24(logm, z, which):
    m_arr, b_arr = _CLARKE[which]
    m = np.interp(z, _CLARKE_ZC, m_arr)
    b = np.interp(z, _CLARKE_ZC, b_arr)
    return m * (logm - 9.16) + b


SFMS_RELATIONS = {
    "speagle14": _speagle14,
    "popesso23": _popesso23,
    "koprowski24": _koprowski24,
    "simmonds25_100myr": lambda logm, z: _simmonds25(logm, z, 0.30, 0.00, 1.06),
    "simmonds25_10myr": lambda logm, z: _simmonds25(logm, z, 0.05, 0.02, 2.30),
    "clarke24_uv": lambda logm, z: _clarke24(logm, z, "uv"),
    "clarke24_ha": lambda logm, z: _clarke24(logm, z, "ha"),
}


def log_sfms(name, logm, z):
    """Analytic log10 SFR of the named relation (no tabulation)."""
    logm = np.asarray(logm, dtype=float)
    return SFMS_RELATIONS[name](logm, z)


# ---------------------------------------------------------------------------
# Tabulated SFMS
# ---------------------------------------------------------------------------

class SFMSTable(object):
    """log10 SFR(log M*, z) on a regular grid with bilinear interpolation.

    ``spec`` is either a relation name or a dict with keys ``logm`` (1D),
    ``z`` (1D) and ``log_sfr`` (2D, shape (len(logm), len(z))) for a custom table.
    """

    def __init__(self, spec):
        if isinstance(spec, dict):
            self.logm = np.asarray(spec["logm"], dtype=float)
            self.z = np.asarray(spec["z"], dtype=float)
            self.table = np.asarray(spec["log_sfr"], dtype=float)
            self.name = spec.get("name", "custom")
        else:
            self.name = spec
            self.logm = LOGM_GRID
            self.z = Z_GRID
            self.table = SFMS_RELATIONS[spec](self.logm[:, None], self.z[None, :])
        self.dlogm = np.diff(self.logm)
        self.dz = np.diff(self.z)

    def __call__(self, logm, z):
        """log10 SFR for an array of log M* at a single redshift z."""
        logm = np.clip(np.asarray(logm, dtype=float), self.logm[0], self.logm[-1])
        z = float(np.clip(z, self.z[0], self.z[-1]))
        j = int(np.clip(np.searchsorted(self.z, z) - 1, 0, len(self.z) - 2))
        fz = (z - self.z[j]) / self.dz[j]
        i = np.clip(np.searchsorted(self.logm, logm) - 1, 0, len(self.logm) - 2)
        fm = (logm - self.logm[i]) / self.dlogm[i]
        col = self.table[:, j] * (1. - fz) + self.table[:, j + 1] * fz
        return col[i] * (1. - fm) + col[i + 1] * fm


def get_sfms_table(spec):
    key = spec if isinstance(spec, str) else id(spec)
    if key not in _SFMS_TABLES:
        _SFMS_TABLES[key] = SFMSTable(spec)
    return _SFMS_TABLES[key]


# ---------------------------------------------------------------------------
# Covariance kernels for the log SFR deviations
# ---------------------------------------------------------------------------

def acf(kernel, dt, p):
    """Autocovariance of log SFR at time lag dt (same units as the timescales in p).

    kernel = "white":         sigma_int^2 delta_ij (independent bins)
             "exp":           sigma_int^2 exp(-|dt|/tau_corr)  (regulator model / damped random walk)
             "ext_regulator": Wan+2024 eq. 3 with sigma_reg, tau_eq, tau_in, sigma_dyn, tau_dyn
    """
    dt = np.abs(dt)
    if kernel == "white":
        return np.where(dt == 0., p["sigma_int"] ** 2, 0.)
    if kernel == "exp":
        return p["sigma_int"] ** 2 * np.exp(-dt / p["tau_corr"])
    if kernel == "ext_regulator":
        t_in, t_eq = p["tau_in"], p["tau_eq"]
        if abs(t_in - t_eq) < 1e-6 * t_in:
            t_eq = t_in * (1. - 1e-6)
        reg = p["sigma_reg"] ** 2 * (t_in * np.exp(-dt / t_in) - t_eq * np.exp(-dt / t_eq)) / (t_in - t_eq)
        return reg + p["sigma_dyn"] ** 2 * np.exp(-dt / p["tau_dyn"])
    raise ValueError("unknown kernel " + str(kernel))


def cholesky_factor(kernel, centres, p):
    """Lower Cholesky factor of the kernel evaluated at the bin centres (cached)."""
    key = (kernel, tuple(np.round(centres, 3)), tuple(sorted((k, round(float(v), 6)) for k, v in p.items())))
    L = _CHOL_CACHE.get(key)
    if L is None:
        c = np.asarray(centres, dtype=float)
        K = acf(kernel, c[:, None] - c[None, :], p)
        K = K + 1e-10 * np.eye(len(c)) * max(np.max(np.diag(K)), 1e-12)
        L = np.linalg.cholesky(K)
        if len(_CHOL_CACHE) >= _CHOL_CACHE_MAX:
            _CHOL_CACHE.clear()
        _CHOL_CACHE[key] = L
    return L


# ---------------------------------------------------------------------------
# Track construction
# ---------------------------------------------------------------------------

def sfms_track(log_mtot, widths_yr, z_centres, delta, table, offset=0., return_frac=0.3,
               quench=None, n_coarse=300, n_fine=60):
    """Build the binned SFH that follows the SFMS and integrates to 10**log_mtot.

    widths_yr, z_centres, delta: per bin, oldest first.
    quench: optional per-bin mean multiplicative factor (same order) applied to the SFR.
    Returns (sfr_bins [Msun/yr, before the quench factor], log_mseed, log_mformed_at_bin_start,
    log_sfms_bins).  The seed mass (stars in place at the start of the oldest bin) is added to
    the oldest bin's SFR.
    """
    widths = np.asarray(widths_yr, dtype=float)
    n = len(widths)
    q = np.ones(n) if quench is None else np.asarray(quench, dtype=float)
    f_surv = 1. - return_frac

    def integrate(log_seed):
        seeds = 10 ** np.asarray(log_seed, dtype=float)
        m = seeds.copy()
        for k in range(n):
            log_sfr = table(np.log10(f_surv * m), z_centres[k]) + offset + delta[k]
            m = m + 10 ** log_sfr * q[k] * widths[k]
        return np.log10(m)

    target = log_mtot
    lo, hi = target - 14., target
    grid = np.linspace(lo, hi, n_coarse)
    out = integrate(grid)
    if out[0] >= target:                       # growth alone exceeds the target: smallest seed
        log_seed = lo
    else:
        j = np.searchsorted(out, target)
        a, b = grid[j - 1], grid[j]
        grid2 = np.linspace(a, b, n_fine)
        out2 = integrate(grid2)
        log_seed = float(np.interp(target, out2, grid2))

    # Final pass storing the per-bin quantities.
    m = 10 ** log_seed
    sfr = np.zeros(n)
    log_m_start = np.zeros(n)
    log_ms = np.zeros(n)
    for k in range(n):
        log_m_start[k] = np.log10(m)
        log_ms[k] = float(table(np.log10(f_surv * m), z_centres[k]))
        sfr[k] = 10 ** (log_ms[k] + offset + delta[k])
        m = m + sfr[k] * q[k] * widths[k]
    sfr[0] += 10 ** log_seed / widths[0] / q[0]
    return sfr, log_seed, log_m_start, log_ms


def lookback_bin_edges(age_of_universe_yr, param):
    """Bin edges in lookback time (yr), ascending, from the component parameters.

    Either ``bin_edges`` (Myr, ascending, last edge clipped to the formation limit) or
    ``n_bins`` log-spaced bins from ``t_first`` (Myr, default 10) to the age of the universe
    at the sampled redshift minus the age at ``z_max`` (default 20).
    """
    z_max = float(param.get("z_max", 20.))
    t_max = age_of_universe_yr - 1e9 * age_at_z(z_max)
    if "bin_edges" in param:
        edges = np.asarray(param["bin_edges"], dtype=float) * 1e6
        edges = edges[edges < t_max]
        edges = np.append(edges, t_max)
    else:
        n = int(param.get("n_bins", 10))
        t_first = float(param.get("t_first", 10.)) * 1e6
        edges = np.concatenate([[0.], np.logspace(np.log10(t_first), np.log10(t_max), n)])
    return edges
