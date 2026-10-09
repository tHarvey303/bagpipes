"""One-page summary figure for a Bagpipes fit: observed spectrum with the posterior model and
residuals, photometry with the posterior SED, star-formation history, surviving- and formed-mass
assembly, and marginal posteriors of the key quantities.

Usage: ``fit.plot_summary()`` or ``bagpipes.plotting.plot_summary(fit)``.  Written to
``pipes/plots/<run>/<ID>_summary.png`` by default.
"""
from __future__ import print_function, division, absolute_import

import os
import numpy as np
import h5py

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
except RuntimeError:
    pass

from .. import utils
from .plot_csfh_posterior import optimize_mah_grid_numpy, MAH_GRID_VERSION
from bagpipes import config

__all__ = ["plot_summary", "add_summary"]

# ---------------------------------------------------------------- style ----
# Conventional journal-figure look: black frames, inward ticks, serif text, no grid.
C = {"data": "black", "data_band": "0.75", "mask": "0.90", "phot": "navy",
     "model": "firebrick", "model_band": "firebrick", "model2": "black", "model2_band": "0.5",
     "ink": "black", "ink2": "0.25", "surface": "white"}
BAND_ALPHA = 0.25
PCT = (16, 50, 84)
RC = {"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
      "mathtext.fontset": "stix", "font.size": 10,
      "axes.linewidth": 0.9, "axes.labelsize": 10, "axes.titlesize": 10,
      "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
      "xtick.minor.visible": True, "ytick.minor.visible": True,
      "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.major.size": 4, "ytick.major.size": 4,
      "xtick.minor.size": 2, "ytick.minor.size": 2, "legend.fontsize": 8.5, "legend.frameon": False,
      "lines.linewidth": 1.0, "axes.grid": False, "text.usetex": False}

LABELS = {  # mathtext only, no LaTeX installation needed
    "stellar_mass": r"$\log_{10}(M_\star/\mathrm{M}_\odot)$",
    "formed_mass": r"$\log_{10}(M_\mathrm{formed}/\mathrm{M}_\odot)$",
    "sfr": r"$\log_{10}(\mathrm{SFR}/\mathrm{M}_\odot\,\mathrm{yr}^{-1})$",
    "sfr_10myr": r"$\log_{10}(\mathrm{SFR}_{10}/\mathrm{M}_\odot\,\mathrm{yr}^{-1})$",
    "ssfr": r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$",
    "mass_weighted_age": r"$\mathrm{age}_\mathrm{MW}$ (Gyr)",
    "tform": r"$t_\mathrm{form}$ (Gyr)",
    "tquench": r"$t_\mathrm{quench}$ (Gyr)",
    "mass_weighted_zmet": r"$Z_\mathrm{MW}/Z_\odot$",
    "redshift": r"$z$",
    "veldisp": r"$\sigma_v$ (km s$^{-1}$)",
    "Av": r"$A_V$ (mag)",
    "metallicity": r"$Z/Z_\odot$",
    "massformed": r"$\log_{10}(M_\mathrm{formed}/\mathrm{M}_\odot)$",
    "tau": r"$\tau$ (Gyr)", "alpha": r"$\alpha$", "beta": r"$\beta$",
    "age": "age (Gyr)", "delta": r"$\delta$", "B": r"$B$", "n": r"$n$",
    "logU": r"$\log U$", "eta": r"$\eta$",
}
DEFAULT_PARAMS = ["stellar_mass", "sfr", "ssfr", "mass_weighted_age", "mass_weighted_zmet",
                  "dust:Av", "redshift", "veldisp"]
LOG_PARAMS = {"sfr", "sfr_10myr"}
FLOORS = {"sfr": -3.0, "sfr_10myr": -3.0, "ssfr": -14.0, "ssfr_10myr": -14.0}  # log10 display floors


def _label(name):
    comp, _, par = name.rpartition(":")
    if name in LABELS:
        return LABELS[name]
    if par in LABELS:
        lab = LABELS[par]
        return lab if comp in ("", "dust") else lab + " [" + comp + "]"
    return name


def _style(ax):
    ax.set_facecolor(C["surface"])
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()


def _to_fnu(wav_AA, f_lambda):
    """erg/s/cm^2/A -> microJy."""
    return f_lambda * wav_AA ** 2 / 2.9979e18 * 1e29


def _fmt(v, lo, hi):
    """Median with asymmetric errors, sensible number of decimals."""
    err = max(abs(hi - v), abs(v - lo), 1e-12)
    nd = int(np.clip(1 - np.floor(np.log10(err)), 0, 4))
    return f"{v:.{nd}f}$^{{+{hi - v:.{nd}f}}}_{{-{v - lo:.{nd}f}}}$"


def _redshift(fit):
    if "redshift" in fit.fitted_model.params:
        return np.median(fit.posterior.samples["redshift"])
    return fit.fitted_model.model_components["redshift"]


def _mah_grid(fit):
    """Surviving-mass assembly grid, cached in the h5 file (same cache as plot_csfh_posterior)."""
    try:
        with h5py.File(fit.fname[:-1] + ".h5", "a") as f:
            if "mah_grid" in f and f["mah_grid"].attrs.get("version", 1) >= MAH_GRID_VERSION:
                return f["mah_grid"][:]
            if "mah_grid" in f:
                del f["mah_grid"]
            mah = optimize_mah_grid_numpy(fit, config)
            d = f.create_dataset("mah_grid", data=mah, compression="gzip", dtype=np.float32)
            d.attrs["version"] = MAH_GRID_VERSION
            return mah
    except OSError:  # h5 locked / read-only: compute without caching
        return optimize_mah_grid_numpy(fit, config)


def _z_axis(ax, zvals):
    ax2 = ax.twiny()
    t = np.interp(zvals, utils.z_array, utils.age_at_z)
    ok = (t > ax.get_xlim()[0]) & (t < ax.get_xlim()[1])
    ax2.set_xticks(t[ok]); ax2.set_xticklabels([str(z) for z in np.asarray(zvals)[ok]])
    ax2.set_xlim(ax.get_xlim()); ax2.tick_params(which="both", direction="in", labelsize=9)
    ax2.minorticks_off()
    ax2.set_xlabel("Redshift")
    return ax2


# ------------------------------------------------------------- panels ----
def _spectrum_panels(fit, ax, axr, units):
    g = fit.galaxy
    wav, flux, err = g.spectrum[:, 0], g.spectrum[:, 1], g.spectrum[:, 2]
    masked = err > 1e20
    spec_post = np.copy(fit.posterior.samples["spectrum"])
    if "calib" in fit.posterior.samples:
        spec_post /= fit.posterior.samples["calib"]
    if "noise" in fit.posterior.samples:
        spec_post += fit.posterior.samples["noise"]
    if units == "fnu":
        conv = wav ** 2 / 2.9979e18 * 1e29
        flux, err, spec_post = flux * conv, err * conv, spec_post * conv
        ylab = r"$f_\nu$ ($\mu$Jy)"
    else:
        sc = 10 ** float(int(np.log10(np.nanmax(flux[~masked]))) - 1)
        flux, err, spec_post = flux / sc, err / sc, spec_post / sc
        ylab = r"$f_\lambda$ (10$^{%d}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)" % int(np.log10(sc))
    post = np.percentile(spec_post, PCT, axis=0)
    fl = np.where(masked, np.nan, flux); er = np.where(masked, np.nan, err)
    ax.fill_between(wav, fl - er, fl + er, color=C["data_band"], alpha=0.5, lw=0, step="mid")
    ax.step(wav, fl, where="mid", color=C["data"], lw=0.8, label="Data")
    ax.fill_between(wav, post[0], post[2], color=C["model_band"], alpha=BAND_ALPHA, lw=0)
    ax.plot(wav, post[1], color=C["model"], lw=1.1, label="Posterior model (16-84 %)")
    # masked regions
    ylo, yhi = np.nanpercentile(fl, 0.5) - 3 * np.nanmedian(er), np.nanpercentile(fl, 99.5) + 3 * np.nanmedian(er)
    yhi = max(yhi, np.nanmax(post[2]) * 1.05)
    ax.set_ylim(max(ylo, -0.2 * yhi), yhi)
    for lo, hi in _runs(wav, masked):
        ax.axvspan(lo, hi, color=C["mask"], lw=0, zorder=0)
    # photometry on the spectrum, if it falls inside the range
    if g.photometry_exists:
        pw, pf, pe = g.filter_set.eff_wavs, g.photometry[:, 1], g.photometry[:, 2]
        ok = (pe < 1e20) & (pw > wav.min()) & (pw < wav.max())
        if units == "fnu":
            pf, pe = _to_fnu(pw, pf), _to_fnu(pw, pe)
        else:
            pf, pe = pf / sc, pe / sc
        ax.errorbar(pw[ok], pf[ok], yerr=pe[ok], fmt="s", ms=5, color=C["phot"], mfc="none", mew=1.0,
                    ecolor=C["phot"], elinewidth=0.9, capsize=2, zorder=5, label="Photometry")
    ax.set_ylabel(ylab)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
    ax.set_xlim(wav.min(), wav.max()); plt.setp(ax.get_xticklabels(), visible=False)
    # residuals of the median model
    res = (flux - post[1]) / err
    res[masked] = np.nan
    axr.axhspan(-1, 1, color="0.85", lw=0); axr.axhline(0, color=C["model"], lw=0.8)
    axr.step(wav, res, where="mid", color=C["data"], lw=0.7)
    for lo, hi in _runs(wav, masked):
        axr.axvspan(lo, hi, color=C["mask"], lw=0, zorder=0)
    lim = min(5, max(3, np.nanpercentile(np.abs(res), 99)))
    axr.set_ylim(-lim, lim); axr.set_xlim(wav.min(), wav.max())
    axr.set_ylabel(r"$\chi$"); axr.set_xlabel(r"Observed wavelength ($\mathrm{\AA}$)")
    nfree = np.sum(~masked)
    return float(np.nansum(res ** 2)), int(nfree)


def _runs(x, mask):
    """Contiguous True runs of mask -> list of (x_start, x_end)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            dx = (x[min(j + 1, n - 1)] - x[max(i - 1, 0)]) / 2
            out.append((x[i] - dx if i > 0 else x[i], x[j] + dx if j < n - 1 else x[j]))
            i = j + 1
        else:
            i += 1
    return out


def _photometry_panel(fit, ax, units):
    g = fit.galaxy
    z = _redshift(fit)
    pw, pf, pe = g.filter_set.eff_wavs, g.photometry[:, 1], g.photometry[:, 2]
    ok = pe < 1e20
    wmod = fit.posterior.model_galaxy.wavelengths * (1. + z)
    sed = np.percentile(fit.posterior.samples["spectrum_full"], PCT, axis=0)
    pmod = np.percentile(fit.posterior.samples["photometry"], PCT, axis=0)
    if units == "fnu":
        conv_m = wmod ** 2 / 2.9979e18 * 1e29; conv_p = pw ** 2 / 2.9979e18 * 1e29
        pf, pe, sed, pmod = pf * conv_p, pe * conv_p, sed * conv_m, pmod * conv_p
        ylab = r"$f_\nu$ ($\mu$Jy)"
    else:
        sc = 10 ** float(int(np.log10(np.nanmax(pf[ok]))) - 1)
        pf, pe, sed, pmod = pf / sc, pe / sc, sed / sc, pmod / sc
        ylab = r"$f_\lambda$ (10$^{%d}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)" % int(np.log10(sc))
    um = 1e-4
    ax.fill_between(wmod * um, sed[0], sed[2], color=C["model_band"], alpha=BAND_ALPHA, lw=0)
    ax.plot(wmod * um, sed[1], color=C["model"], lw=0.8, label="Posterior SED")
    ax.plot(pw[ok] * um, pmod[1][ok], "o", ms=7, mfc="none", mec=C["model"], mew=1.0, label="Model photometry")
    det = ok & (pf > 2 * pe); lim = ok & ~det
    ax.errorbar(pw[det] * um, pf[det], yerr=pe[det], fmt="s", ms=5, color=C["phot"], mfc="none", mew=1.0,
                ecolor=C["phot"], elinewidth=0.9, capsize=2, zorder=5, label="Photometry")
    if lim.any():   # 2-sigma upper limits for bands with S/N < 2
        ax.plot(pw[lim] * um, 2 * pe[lim], "v", ms=6, color=C["phot"], mfc="none", mew=1.0, zorder=5, label=r"2$\sigma$ upper limit")
    lo = max(pw[ok].min() * um * 0.8, wmod.min() * um); hi = min(pw[ok].max() * um * 1.25, wmod.max() * um)
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    if units == "fnu":
        ymin = max(np.nanmin(np.r_[pf[det] - pe[det], pmod[1][ok], 2 * pe[lim]]) * 0.3, np.nanmax(pf[ok]) * 1e-3)
        ax.set_yscale("log"); ax.set_ylim(ymin, np.nanmax(np.r_[pf[ok] + pe[ok], pmod[2][ok]]) * 2)
    else:
        ax.set_ylim(0, np.nanmax(np.r_[pf[ok] + pe[ok], pmod[2][ok]]) * 1.2)
    ax.set_xlabel(r"Observed wavelength ($\mu$m)"); ax.set_ylabel(ylab)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.legend(loc="lower right")
    chi2 = float(np.median(fit.posterior.samples["chisq_phot"])) if "chisq_phot" in fit.posterior.samples else np.nan
    return chi2, int(ok.sum())


def _sfh_panel(fit, ax, zvals):
    z = _redshift(fit)
    t_obs = np.interp(z, utils.z_array, utils.age_at_z)
    t = t_obs - fit.posterior.sfh.ages * 1e-9
    post = np.percentile(fit.posterior.samples["sfh"], PCT, axis=0)
    ax.fill_between(t, post[0], post[2], color=C["model_band"], alpha=BAND_ALPHA, lw=0)
    ax.plot(t, post[1], color=C["model"], lw=1.2)
    for key, ls in (("tform", "--"), ("tquench", ":")):
        if key in fit.posterior.samples:
            v = fit.posterior.samples[key]; v = v[v < 90]
            if len(v):
                ax.axvline(np.median(v), color="black", lw=0.8, ls=ls)
                ax.text(np.median(v), 0.96, " " + key.replace("t", "$t_\\mathrm{") + "}$", transform=ax.get_xaxis_transform(),
                        fontsize=8, va="top")
    ax.set_xlim(0, t_obs * 1.02); ax.set_ylim(0, max(post[2].max() * 1.15, 1e-3))
    ax.set_xlabel("Age of the Universe (Gyr)"); ax.set_ylabel(r"SFR (M$_\odot$ yr$^{-1}$)")
    _z_axis(ax, zvals)
    return t, t_obs


def _mass_panel(fit, ax, t, t_obs, zvals):
    widths = fit.posterior.sfh.age_widths
    sfh = fit.posterior.samples["sfh"]
    formed = np.cumsum((sfh * widths)[:, ::-1], axis=1)[:, ::-1]   # mass formed before each lookback age
    pf = np.percentile(np.log10(np.clip(formed, 1., None)), PCT, axis=0)
    ax.fill_between(t, pf[0], pf[2], color=C["model2_band"], alpha=0.2, lw=0)
    ax.plot(t, pf[1], color=C["model2"], lw=1.0, ls="--", label="Formed")
    try:
        mah = _mah_grid(fit)
        pm = np.percentile(np.log10(np.clip(mah, 1., None)), PCT, axis=0)
        ax.fill_between(t, pm[0], pm[2], color=C["model_band"], alpha=BAND_ALPHA, lw=0)
        ax.plot(t, pm[1], color=C["model"], lw=1.2, label="Surviving")
        top = max(pm[2].max(), pf[2].max())
    except Exception as e:  # pragma: no cover
        print("plot_summary: surviving-mass track skipped:", e); top = pf[2].max()
    ax.set_xlim(0, t_obs * 1.02); ax.set_ylim(top - 3.5, top + 0.3)
    ax.set_xlabel("Age of the Universe (Gyr)"); ax.set_ylabel(r"$\log_{10}(M/\mathrm{M}_\odot)$")
    ax.legend(loc="lower right")
    _z_axis(ax, zvals)


def _marginal(ax, samples, name):
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    key = name.split(":")[-1]
    if key in LOG_PARAMS:
        x = np.log10(np.clip(x, 1e-30, None))
    floor = FLOORS.get(key)
    if floor is not None:
        x = np.clip(x, floor, None)
    lo, med, hi = np.percentile(x, PCT)
    at_floor = floor is not None and lo <= floor + 1e-6
    width = max(x.max() - x.min(), 1e-6)
    rng = (x.min() - 0.05 * width, x.max() + 0.05 * width)
    if at_floor:
        rng = (floor - 0.05 * width, x.max() + 0.05 * width)
    y, edges = np.histogram(x, bins=30, range=rng, density=True)
    ax.hist(x, bins=edges, density=True, histtype="stepfilled", color="0.85", edgecolor="black", lw=0.8)
    for v in (lo, med, hi):
        ax.axvline(v, color="black", lw=0.8, ls="--")
    ax.set_yticks([]); ax.set_ylim(0, y.max() * 1.15); ax.set_xlim(rng)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    title = f"$<$ {hi:.2f} (84%)" if at_floor else _fmt(med, lo, hi)
    ax.set_xlabel(_label(name)); ax.set_title(title)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))


# --------------------------------------------------------------- main ----
def add_summary(fit, fig=None, units="fnu", params=None, zvals=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20)):
    """Draw the summary figure for ``fit`` into ``fig`` (created if None).  Returns fig."""
    fit.posterior.get_advanced_quantities()
    g = fit.galaxy
    has_spec, has_phot = g.spectrum_exists, g.photometry_exists
    params = [p for p in (params or DEFAULT_PARAMS) if p in fit.posterior.samples] or [fit.fitted_model.params[0]]
    n_marg = len(params); n_mcol = min(4, n_marg); n_mrow = int(np.ceil(n_marg / n_mcol))

    if fig is None:
        fig = plt.figure(figsize=(15, 9.5 + 1.9 * (n_mrow - 1)))
    fig.patch.set_facecolor(C["surface"])
    heights = [2.4, 0.7, 2.0] + [1.3] * n_mrow
    gs = gridspec.GridSpec(len(heights), 12, figure=fig, height_ratios=heights, hspace=0.75, wspace=0.9,
                           left=0.055, right=0.985, top=0.9, bottom=0.07)
    chi2_s = chi2_p = nfree_s = nphot = None
    if has_spec and has_phot:
        ax_s = fig.add_subplot(gs[0, :8]); ax_r = fig.add_subplot(gs[1, :8], sharex=ax_s); ax_p = fig.add_subplot(gs[0:2, 8:])
    elif has_spec:
        ax_s = fig.add_subplot(gs[0, :]); ax_r = fig.add_subplot(gs[1, :], sharex=ax_s); ax_p = None
    else:
        ax_p = fig.add_subplot(gs[0:2, :]); ax_s = ax_r = None
    for a in (ax_s, ax_r, ax_p):
        if a is not None:
            _style(a)
    if ax_s is not None:
        chi2_s, nfree_s = _spectrum_panels(fit, ax_s, ax_r, units)
    if ax_p is not None:
        chi2_p, nphot = _photometry_panel(fit, ax_p, units)

    ax_sfh = fig.add_subplot(gs[2, :6]); ax_m = fig.add_subplot(gs[2, 6:])
    _style(ax_sfh); _style(ax_m)
    t, t_obs = _sfh_panel(fit, ax_sfh, zvals)
    _mass_panel(fit, ax_m, t, t_obs, zvals)

    for i, p in enumerate(params):
        ax = fig.add_subplot(gs[3 + i // n_mcol, (i % n_mcol) * 3:(i % n_mcol) * 3 + 3])
        _style(ax); _marginal(ax, fit.posterior.samples[p], p)

    # header
    z = _redshift(fit)
    bits = [f"{g.ID}", f"run: {fit.run}", f"z = {z:.4f}"]
    if "lnz" in fit.results:
        bits.append(f"ln Z = {float(fit.results['lnz']):.1f}")
    if chi2_s is not None:
        bits.append(f"spectrum: {nfree_s} px, $\\chi^2_\\nu$(median model) = {chi2_s / max(nfree_s, 1):.2f}")
    if chi2_p is not None and np.isfinite(chi2_p):
        bits.append(f"photometry: {nphot} bands, $\\chi^2$ = {chi2_p:.1f}")
    bits.append(f"{fit.posterior.n_samples} posterior samples")
    fig.text(0.055, 0.965, bits[0], fontsize=13, va="center")
    fig.text(0.055, 0.937, ";   ".join(bits[1:]), fontsize=9, va="center")
    return fig


def plot_summary(fit, show=False, save=True, units="fnu", params=None, fname=None, dpi=150, fmt="png",
                 zvals=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20)):
    """One-page summary of a fit: data + posterior model (spectrum with residuals, photometry with the
    posterior SED), SFH, mass assembly (formed and surviving), and marginal posteriors.

    units : "fnu" (microJy, default) or "flambda".
    params : list of sample keys for the marginal panels (default: stellar_mass, sfr, ssfr,
             mass_weighted_age, mass_weighted_zmet, dust:Av, redshift, veldisp, where available).
    """
    with mpl.rc_context(RC):
        fig = add_summary(fit, units=units, params=params, zvals=zvals)
    if save:
        if fname is None:
            d = os.path.join("pipes", "plots", fit.run)
            os.makedirs(d, exist_ok=True)
            fname = os.path.join(d, f"{fit.galaxy.ID}_summary.{fmt}")
        fig.savefig(fname, dpi=dpi, facecolor=fig.get_facecolor())
        if not show:
            plt.close(fig)
    if show:
        plt.show(); plt.close(fig)
    return fig
