"""
Refactored GP fitting module with a shared base class and backend-specific subclasses.

Classes
-------
GPFit : Base class with all shared functionality (data loading, plotting, priors, etc.)
CeleriteGPFit(GPFit) : celerite2 backend (PyMC + NumPyro samplers)
TinygpGPFit(GPFit) : tinygp/JAX backend (NumPyro sampler, GPU-compatible)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import corner

from .utils import estimate_inverse_gamma_parameters
from .masking import get_object_params
from . import config

paths = config.paths

VAR_NAMES = [
    "mean", "period", "log_jitter",
    "log_sigma", "log_rho", "log_sigma_rot",
    "log_Q0", "log_dQ", "f",
]

__all__ = ["GPFit", "VAR_NAMES", "WalkerPlotCallback"]


# ------------------------------------------------------------------
# WalkerPlotCallback
# ------------------------------------------------------------------

class WalkerPlotCallback:
    """Live trace plot updated during PyMC sampling."""

    def __init__(self, var_names, save_path, update_every=50, ylims=None, priors=None,
                 map_soln=None, prior_params=None):
        self.var_names = var_names
        self.save_path = save_path
        self.update_every = update_every
        self._chain_data = {}
        self.ylims = ylims or {}
        self.priors = priors or {}
        self.map_soln = map_soln or {}
        self.prior_params = prior_params or {}

        plt.ion()
        n = len(var_names)
        self.fig, self.axes = plt.subplots(
            n, 2, figsize=(16, 3 * n),
            sharey='row',
            squeeze=False,
            gridspec_kw={'width_ratios': [3, 1], 'hspace': 0, 'wspace': 0.},
        )
        self.fig.tight_layout(h_pad=0)

        for i, name in enumerate(self.var_names):
            if name in self.ylims:
                self.axes[i, 0].set_ylim(self.ylims[name])

            self.axes[i, 1].cla()
            self.axes[i, 1].tick_params(axis='y', labelleft=False)
            if name in self.priors:
                dist = self.priors[name]
                if name in self.ylims:
                    lo, hi = self.ylims[name]
                else:
                    try:
                        lo, hi = float(dist.ppf(0.001)), float(dist.ppf(0.999))
                    except Exception:
                        m, s = float(dist.mean()), float(dist.std())
                        lo, hi = m - 4 * s, m + 4 * s
                y_vals = np.linspace(lo, hi, 300)
                pdf_vals = dist.pdf(y_vals)
                self.axes[i, 1].plot(pdf_vals, y_vals, color='C0', lw=1.5)
                self.axes[i, 1].fill_betweenx(y_vals, 0, pdf_vals, alpha=0.2, color='C0')
                self.axes[i, 1].set_xlim(left=0)
            if self.map_soln and name in self.map_soln:
                self.axes[i, 1].axhline(float(self.map_soln[name]), color='k', ls='--', lw=1.5)

            if self.prior_params and name in self.prior_params:
                dist_text = self._format_prior_label(self.prior_params[name])
                self.axes[i, 1].text(1.04, 0.25, dist_text, transform=self.axes[i, 1].transAxes,
                                     va="center", ha="left", fontsize=20, family="monospace", linespacing=1.6)
            if self.map_soln and name in self.map_soln:
                map_text = f"MAP = {float(self.map_soln[name]):.4g}"
                self.axes[i, 1].text(1.04, 0.5, map_text, transform=self.axes[i, 1].transAxes,
                                     va="center", ha="left", fontsize=20, family="monospace", linespacing=1.6)

    @staticmethod
    def _format_prior_label(spec):
        """Return a short human-readable string for a prior spec dict."""
        t = spec["type"]
        if t == "normal":
            return f"Normal\nμ = {spec['mu']:.3g}\nσ = {spec['sigma']:.3g}"
        if t == "truncated_normal":
            return f"TruncNormal\nμ = {spec['mu']:.3g}\nσ = {spec['sigma']:.3g}\n[{spec['lower']:.3g},  {spec['upper']:.3g}]"
        if t == "uniform":
            return f"Uniform\n[{spec['lower']:.3g},  {spec['upper']:.3g}]"
        if t == "half_normal":
            return f"HalfNormal\nσ = {spec['sigma']:.3g}"
        if t == "log_normal":
            return f"LogNormal\nμ = {spec['mu']:.3g}\nσ = {spec['sigma']:.3g}"
        if t == "inverse_gamma":
            return f"InvGamma\n[{spec['lower']:.3g},  {spec['upper']:.3g}]"
        if t == "fixed":
            return f"Fixed\n{spec['value']:.3g}"
        return t

    def __call__(self, trace, draw):
        if draw.draw_idx == 0 or draw.draw_idx % self.update_every != 0:
            return

        chain = draw.chain
        chain_vals = {}
        for name in self.var_names:
            try:
                vals = trace.get_values(name)[:draw.draw_idx + 1]
                chain_vals[name] = np.atleast_1d(vals).reshape(len(vals), -1)
            except Exception:
                return
        self._chain_data[chain] = chain_vals

        for i, name in enumerate(self.var_names):
            self.axes[i, 0].cla()
            for c in sorted(self._chain_data):
                if name in self._chain_data[c]:
                    self.axes[i, 0].plot(self._chain_data[c][name], lw=1)
            self.axes[i, 0].set_ylabel(name, fontsize=25)
        self.axes[-1, 0].set_xlabel("draw", fontsize=25)

        self.fig.savefig(self.save_path, bbox_inches="tight")
        plt.pause(0.001)


# ==================================================================
# GPFit base class
# ==================================================================

class GPFit:
    """GP rotation-period fitter for a single TESS target (base class).

    Subclass and override the backend-specific methods (marked with
    ``NotImplementedError``) to use a particular GP library.

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    masked_h5_path : str, optional
        Path to the masked HDF5 file. Defaults to paths['MASK_LC_PATH']/TIC{tic_id}.h5.
    output_dir : str, optional
        Base directory for outputs. Defaults to paths['RESULTS_PATH'].
    catalog_path : str, optional
        Path to the EB parameter catalog CSV.
    bin_size : float, optional
        If given, bin the data to this cadence in days before fitting.
    """

    var_names = VAR_NAMES

    @property
    def free_var_names(self):
        """Sampled (non-fixed) variable names."""
        if self._prior_params is None:
            return list(self.var_names)
        return [name for name, spec in self._prior_params.items() if spec["type"] != "fixed"]

    def __init__(self, tic_id, masked_h5_path=None, output_dir=None,
                 catalog_path=None, bin_size=None, downsample=1):
        self.tic_id = str(tic_id)

        if masked_h5_path is None:
            masked_h5_path = os.path.join(paths["MASK_LC_PATH"], f"TIC{self.tic_id}.h5")
        self.masked_h5_path = masked_h5_path

        if output_dir is None:
            output_dir = paths["RESULTS_PATH"]
        self.output_dir = output_dir

        self.catalog_path = catalog_path
        self.bin_size = bin_size

        params = get_object_params(self.tic_id, catalog_path)
        self.prot_init = params["Prot"]

        self.sector = None
        self.x = None
        self.y = None
        self.yerr = None
        self.x_transit = None
        self.y_transit = None
        self.trace = None
        self.summary = None
        self.map_soln = None
        self.flux_factor = None
        self.downsample = downsample
        self._prior_params = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, sector):
        """Load rotation-masked lightcurve for one or more sectors."""
        sectors = [sector] if np.ndim(sector) == 0 else list(sector)
        self.sector = sector

        x_all, y_all, yerr_all = [], [], []
        xt_all, yt_all = [], []
        for s in sectors:
            xs, ys, yerrs = self._load_rotation_masked(s)
            x_all.append(xs)
            y_all.append(ys)
            yerr_all.append(yerrs)
            self._load_transit_points(s)
            xt_all.append(self.x_transit)
            yt_all.append(self.y_transit)

        self.x = np.concatenate(x_all)
        self.y = np.concatenate(y_all)
        self.yerr = np.concatenate(yerr_all)
        self.x_transit = np.concatenate(xt_all)
        self.y_transit = np.concatenate(yt_all)

        order = np.argsort(self.x)
        self.x, self.y, self.yerr = self.x[order], self.y[order], self.yerr[order]
        order_t = np.argsort(self.x_transit)
        self.x_transit = self.x_transit[order_t]
        self.y_transit = self.y_transit[order_t]

        self.map_soln = None
        self._model = None
        self.trace = None
        self.summary = None
        self.binned_cadence = np.median(np.diff(self.x))
        sector_label = self.sector_label
        print(f"Loaded data for TIC {self.tic_id} sector {sector_label}: "
              f"{len(self.x)} points (binned: {self.bin_size is not None} | downsampled: {self.downsample > 1})")
        print(f"Time span: {self.x.max() - self.x.min():.2f} days")
        print(f"Raw cadence: {self.raw_cadence:.4f} days | {self.raw_cadence*24*60:.2f} minutes")
        print(f"Binned cadence: {self.binned_cadence:.4f} days | {self.binned_cadence*24*60:.2f} minutes")
        print(f"Initial Prot estimate: {self.prot_init:.2f} days\n")
        return self.x, self.y, self.yerr

    @property
    def sector_label(self):
        """Return a display-friendly string for the current sector(s)."""
        if isinstance(self.sector, (list, tuple)):
            return "+".join(str(s) for s in self.sector)
        return str(self.sector)

    def _load_rotation_masked(self, sector):
        """Load rotation-masked lightcurve from HDF5."""
        with h5py.File(self.masked_h5_path, "r") as f:
            grp = f[f"rotation_masked/sector{sector}"]
            x = grp["time"][:]
            y = grp["flux"][:]
            yerr = grp["flux_err"][:]

        idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (y > 0)
        x, y, yerr = x[idx], y[idx], yerr[idx]
        self.raw_cadence = np.median(np.diff(x))

        if self.bin_size is not None and self.bin_size > 0:
            x, y, yerr = self._bin_data(x, y, yerr, self.bin_size)

        if self.downsample > 1:
            x = x[::self.downsample]
            y = y[::self.downsample]
            yerr = yerr[::self.downsample]

        x -= x.min()
        mu = np.mean(y)
        if self.flux_factor is None:
            self.flux_factor = 1 / np.std(y)

        y = (y / mu - 1) * self.flux_factor
        yerr = yerr / mu * self.flux_factor

        return (
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(yerr, dtype=np.float64),
        )

    def _load_transit_points(self, sector):
        """Load transit-masked points for plotting."""
        with h5py.File(self.masked_h5_path, "r") as f:
            tgrp = f[f"transit_masked/sector{sector}"]
            xt = tgrp["time"][:]
            yt = tgrp["flux"][:]
            rgrp = f[f"rotation_masked/sector{sector}"]
            x_raw_min = rgrp["time"][:].min()
            y_raw = rgrp["flux"][:]
            y_raw_mean = np.nanmean(y_raw[np.isfinite(y_raw) & (y_raw > 0)])
        idx_t = np.isfinite(xt) & np.isfinite(yt) & (yt > 0)
        y_raw_std = np.std(y_raw[np.isfinite(y_raw) & (y_raw > 0)])
        self.x_transit = xt[idx_t] - x_raw_min
        self.y_transit = (yt[idx_t] / y_raw_mean - 1) / y_raw_std

    @staticmethod
    def _bin_data(x, y, yerr, bin_size):
        """Bin time-series data into uniform time bins."""
        bins = np.arange(x.min(), x.max() + bin_size, bin_size)
        bin_indices = np.digitize(x, bins)

        x_bin, y_bin, yerr_bin = [], [], []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            x_bin.append(np.mean(x[mask]))
            y_bin.append(np.mean(y[mask]))
            yerr_bin.append(np.sqrt(np.sum(yerr[mask] ** 2)) / mask.sum())

        return np.array(x_bin), np.array(y_bin), np.array(yerr_bin)

    # ------------------------------------------------------------------
    # Resolve helper
    # ------------------------------------------------------------------

    def _resolve_from_soln(self, soln):
        """Return a resolve(canonical) callable for a parameter solution dict.

        Checks ``log_{canonical}`` first (→ exp), then ``canonical``
        directly, then falls back to a fixed value in _prior_params.
        """
        s = soln
        p = self._prior_params or {}

        def resolve(canonical):
            log_name = f"log_{canonical}"
            if log_name in s:
                return float(np.exp(s[log_name]))
            if canonical in s:
                return float(s[canonical])
            if log_name in p and p[log_name]["type"] == "fixed":
                return float(np.exp(p[log_name]["value"]))
            if canonical in p and p[canonical]["type"] == "fixed":
                return float(p[canonical]["value"])
            raise KeyError(f"Parameter {canonical!r} not found in soln or _prior_params")

        return resolve

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def _prepare_init_points(self, map_soln, chains, init_jitter=0.1):
        """Clamp MAP solution inside bounds and create per-chain jittered starts."""
        import pandas as pd

        if map_soln is None:
            return None
        if isinstance(map_soln, pd.Series):
            map_soln = {k: float(map_soln[k]) for k in self.free_var_names if k in map_soln.index}

        if isinstance(map_soln, dict) and self._prior_params:
            eps = 1e-3
            for name, spec in self._prior_params.items():
                if spec["type"] in ("uniform", "truncated_normal") and name in map_soln:
                    span = spec["upper"] - spec["lower"]
                    lo = spec["lower"] + eps * span
                    hi = spec["upper"] - eps * span
                    map_soln[name] = float(np.clip(map_soln[name], lo, hi))

        if not (isinstance(map_soln, dict) and map_soln):
            return None

        rng = np.random.default_rng()
        initvals = []
        for _ in range(chains):
            jittered = {}
            for k, v in map_soln.items():
                scale = init_jitter * abs(v) if v != 0 else init_jitter
                jittered[k] = v + rng.normal(0, scale)
            if self._prior_params:
                for name, spec in self._prior_params.items():
                    if spec["type"] in ("uniform", "truncated_normal") and name in jittered:
                        eps = 1e-3
                        span = spec["upper"] - spec["lower"]
                        lo = spec["lower"] + eps * span
                        hi = spec["upper"] - eps * span
                        jittered[name] = float(np.clip(jittered[name], lo, hi))
            initvals.append(jittered)
        return initvals

    def _sample_numpyro(self, model_fn, tune, draws, chains, target_accept,
                        chain_method="parallel", map_soln=None, step_scale=None,
                        init_jitter=0.1):
        """Run MCMC sampling with the NumPyro NUTS sampler."""
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.infer
        from jax import random as jax_random

        initvals = self._prepare_init_points(map_soln, chains, init_jitter)

        init_params = None
        if initvals is not None:
            init_params = {
                k: jnp.array([d[k] for d in initvals])
                for k in initvals[0]
            }

        nuts_kwargs = dict(target_accept_prob=target_accept)
        if step_scale is not None:
            nuts_kwargs["step_size"] = step_scale

        kernel = numpyro.infer.NUTS(model_fn, **nuts_kwargs)
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            chain_method=chain_method,
        )
        mcmc.run(jax_random.PRNGKey(0), init_params=init_params)

        trace = az.from_numpyro(mcmc)
        return trace

    # ------------------------------------------------------------------
    # Saving results
    # ------------------------------------------------------------------

    def _save_results(self):
        """Save trace and summary statistics to disk."""
        sector_tag = f"TIC{self.tic_id}_sector{self.sector_label}"

        posterior_dir = os.path.join(self.output_dir, f"posterior_results/TIC{self.tic_id}")
        os.makedirs(posterior_dir, exist_ok=True)
        self.trace.to_netcdf(os.path.join(posterior_dir, f"{sector_tag}.nc"))

        self.summary = az.summary(self.trace, var_names=self.free_var_names)
        summary_dir = os.path.join(self.output_dir, f"summary_stats/TIC{self.tic_id}")
        os.makedirs(summary_dir, exist_ok=True)
        self.summary.to_csv(os.path.join(summary_dir, f"{sector_tag}.csv"))

    # ------------------------------------------------------------------
    # Reload saved results
    # ------------------------------------------------------------------

    def reload(self, sector, results_dir=None):
        """Reload saved GP fit results for a given sector."""
        import pandas as pd

        if results_dir is None:
            results_dir = self.output_dir

        self.sector = sector
        sector_label = self.sector_label
        sector_tag = f"TIC{self.tic_id}_sector{sector_label}"

        trace_path = os.path.join(results_dir, "posterior_results", f"{sector_tag}.nc")
        self.trace = az.from_netcdf(trace_path)

        summary_path = os.path.join(results_dir, "summary_stats", f"{sector_tag}.csv")
        self.summary = pd.read_csv(summary_path, index_col=0)

        self.load_data(sector)
        return self

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def _get_priors(self):
        """Return scipy prior distributions for each parameter."""
        from scipy import stats

        if self._prior_params is None:
            raise ValueError("No prior parameters defined. Set self._prior_params first.")

        priors = {}
        for name, p in self._prior_params.items():
            if p["type"] == "normal":
                priors[name] = stats.norm(loc=p["mu"], scale=p["sigma"])
            elif p["type"] == "truncated_normal":
                a = (p["lower"] - p["mu"]) / p["sigma"]
                b = (p["upper"] - p["mu"]) / p["sigma"]
                priors[name] = stats.truncnorm(a, b, loc=p["mu"], scale=p["sigma"])
            elif p["type"] == "uniform":
                priors[name] = stats.uniform(loc=p["lower"], scale=p["upper"] - p["lower"])
            elif p["type"] == "half_normal":
                priors[name] = stats.halfnorm(scale=p["sigma"])
            elif p["type"] == "log_normal":
                priors[name] = stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
            elif p["type"] == "inverse_gamma":
                ig = estimate_inverse_gamma_parameters(p["lower"], p["upper"])
                priors[name] = stats.invgamma(a=ig["alpha"], scale=ig["beta"])
        return priors

    def get_walker_ylims(self, sigma_factor=5):
        """Return y-axis limits for ``WalkerPlotCallback`` based on the priors."""
        priors = self._get_priors()

        ylims = {}
        for name, p in self._prior_params.items():
            if p["type"] == "uniform":
                ylims[name] = (p["lower"], p["upper"])
            elif p["type"] == "truncated_normal":
                ylims[name] = (p["lower"], p["upper"])
            elif p["type"] == "normal":
                ylims[name] = (p["mu"] - sigma_factor * p["sigma"], p["mu"] + sigma_factor * p["sigma"])
            elif name in priors:
                dist = priors[name]
                ylims[name] = (float(dist.ppf(0.001)), float(dist.ppf(0.999)))
        return ylims

    # ------------------------------------------------------------------
    # PSD / ACF computation
    # ------------------------------------------------------------------

    def compute_psd(self, freq_min=None, freq_max=None, n_freq=5000, log_spacing=True):
        """Compute the Lomb-Scargle power spectral density of the lightcurve."""
        from astropy.timeseries import LombScargle

        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        dt = np.median(np.diff(self.x))
        T = self.x.max() - self.x.min()
        if freq_min is None:
            freq_min = 1.0 / T
        if freq_max is None:
            freq_max = 0.5 / dt

        if log_spacing:
            freq = 10**np.linspace(np.log10(freq_min), np.log10(freq_max), n_freq)
        else:
            freq = np.linspace(freq_min, freq_max, n_freq)
        period = 1.0 / freq[::-1]

        ls = LombScargle(self.x, self.y)
        power = ls.power(freq, normalization="psd")
        power_vs_period = power[::-1]

        return freq, period, power, power_vs_period

    # ------------------------------------------------------------------
    # Backend-specific methods (must be overridden by subclasses)
    # ------------------------------------------------------------------

    def build_pymc_model(self, prior_params=None):
        raise NotImplementedError

    def build_numpyro_model(self, prior_params=None):
        raise NotImplementedError

    def find_map(self, sector, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def _build_kernel(self, resolve, which="full"):
        raise NotImplementedError

    def predict(self, xgrid=None):
        raise NotImplementedError

    def predict_map(self, xgrid=None):
        raise NotImplementedError

    # def plot_kernel_components(self, output_dir=None, soln=None):
    #     raise NotImplementedError

    def _draw_psd(self, ax, soln=None, show_kernel=True,
                  freq_min=None, freq_max=None, n_freq=5000, log_spacing=False):
        raise NotImplementedError

    def _draw_acf(self, ax, soln=None, show_kernel=True, max_lag=None,
                  n_lags=500):
        raise NotImplementedError

    def fit_kernel_initial(self, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Plotting (shared — these call backend-specific predict/predict_map)
    # ------------------------------------------------------------------

    def plot_raw(self, sector=None, output_dir=None):
        """Plot the raw lightcurve for a sector showing rotation and transit data."""
        if sector is None:
            sector = self.sector
        if sector is None:
            raise RuntimeError("No sector specified. Pass sector or call load_data first.")
        sectors = [sector] if np.ndim(sector) == 0 else list(sector)
        sector_label = "+".join(str(s) for s in sectors) if len(sectors) > 1 else str(sectors[0])

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"raw_lc_plots/TIC{self.tic_id}")

        x_rot_all, y_rot_all, yerr_rot_all = [], [], []
        x_tra_all, y_tra_all = [], []
        with h5py.File(self.masked_h5_path, "r") as f:
            for s in sectors:
                rgrp = f[f"rotation_masked/sector{s}"]
                xr = rgrp["time"][:]
                yr = rgrp["flux"][:]
                yerr_r = rgrp["flux_err"][:]
                idx = np.isfinite(xr) & np.isfinite(yr) & (yr > 0)
                x_rot_all.append(xr[idx])
                y_rot_all.append(yr[idx])
                yerr_rot_all.append(yerr_r[idx])

                tra_key = f"transit_masked/sector{s}"
                if tra_key in f:
                    tgrp = f[tra_key]
                    xt = tgrp["time"][:]
                    yt = tgrp["flux"][:]
                    idx_t = np.isfinite(xt) & np.isfinite(yt) & (yt > 0)
                    x_tra_all.append(xt[idx_t])
                    y_tra_all.append(yt[idx_t])

        x_rot = np.concatenate(x_rot_all)
        y_rot = np.concatenate(y_rot_all)
        yerr_rot = np.concatenate(yerr_rot_all)
        x_tra = np.concatenate(x_tra_all) if x_tra_all else None
        y_tra = np.concatenate(y_tra_all) if y_tra_all else None

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.errorbar(x_rot, y_rot, yerr=yerr_rot, fmt=".k", capsize=0,
                    ms=2, alpha=0.8, label="Rotation data")
        if x_tra is not None and y_tra is not None:
            ax.plot(x_tra, y_tra, ".C1", ms=3, alpha=0.6, label="Transit points")

        ax.set_xlabel("Time [JD]", fontsize=16)
        ax.set_ylabel("Flux", fontsize=16)
        ax.set_title(f"TIC {self.tic_id} — Sector {sector_label} (raw)", fontsize=18)
        ax.legend(fontsize=12)
        ax.minorticks_on()

        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{sector_label}_raw_lc.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved kernel-component plot → {fname}\n")

    def plot_map(self, output_dir=None, return_fig=False, alpha=0.2):
        """Plot the lightcurve with the MAP GP prediction overlaid."""
        if self.map_soln is None:
            raise RuntimeError("No MAP solution available. Run find_map() first.")

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"gp_map_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)

        xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
        mu_map, var_map = self.predict_map(xgrid=xgrid)
        sigma_map = np.sqrt(var_map)

        if self.x_transit is None or self.y_transit is None:
            self.load_data(self.sector)

        fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=False, sharey=True)

        if "period" in self.map_soln:
            prot_map = self.map_soln["period"]
        elif "log_period" in self.map_soln:
            prot_map = np.exp(self.map_soln["log_period"])
        else:
            prot_map = None

        for ax in axs:
            ax.errorbar(self.x, self.y, yerr=self.yerr, fmt=".k", capsize=0, label=f"Rotation data (N={len(self.x)} | cadence={self.binned_cadence*24*60:.1f} min)")
            ax.plot(self.x_transit - (self.bin_size or 0), self.y_transit, ".C1", ms=4, alpha=0.6, label=f"Transit points")
            ax.plot(xgrid, mu_map, color="b", lw=1.5, label=f"MAP prediction (Prot={prot_map:.2f} d)")
            ax.fill_between(xgrid, mu_map - sigma_map, mu_map + sigma_map, color="b", alpha=alpha)
            ax.set_ylabel("Relative Flux", fontsize=20)
            ax.minorticks_on()

        axs[0].legend(fontsize=12)

        xspan = (self.x.max() - self.x.min()) / 2
        yspan = np.nanstd(self.y)
        axs[0].set_xlim(self.x.min(), self.x.min() + xspan)
        axs[0].set_ylim(self.y.mean() - 3 * yspan, self.y.mean() + 3 * yspan)
        axs[0].set_title(f"TIC {self.tic_id} — Sector {self.sector_label} (MAP)", fontsize=22)

        axs[1].set_xlim(max(self.x.min(), self.x.max() - xspan), self.x.max())
        axs[1].set_ylim(self.y.mean() - 3 * yspan, self.y.mean() + 3 * yspan)
        axs[1].set_xlabel("Time [days]", fontsize=20)

        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_map_fit.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        if return_fig:
            return fig, axs
        plt.close(fig)
        print(f"Saved kernel-component plot → {fname}\n")

    def plot_gp_fit(self, xgrid, mu, var, output_dir=None, mu_map=None, var_map=None):
        """Plot the MAP fit with the full GP posterior overlaid."""
        sigma = np.sqrt(var)

        fig, axs = self.plot_map(output_dir=output_dir, return_fig=True, alpha=0.1)

        for ax in axs:
            ax.plot(xgrid, mu, color="r", label=f"GP posterior mean (Prot={float(self.summary.loc['period', 'mean']):.2f} d)")
            ax.fill_between(xgrid, mu - sigma, mu + sigma, color="r", alpha=0.3)

        axs[0].legend(fontsize=12)
        axs[0].set_title(f"TIC {self.tic_id} — Sector {self.sector_label}", fontsize=22)

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"gp_fit_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)

        fig.tight_layout()
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_gp_fit.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved GP fit plot → {fname}\n")
        
    def _draw_kernel_component_figure(self, predictions, colors, soln, xgrid, output_dir):
        """Shared kernel component figure layout (used by both backends)."""
        s = soln
        try:
            resolve = self._resolve_from_soln(soln)
            prot_map = resolve("period")
        except KeyError:
            prot_map = None
        n_panels = len(predictions)
        if n_panels == 0:
            print("No kernel components to plot.")
            return

        fig, axs = plt.subplots(n_panels, 1, figsize=(20, 14 * n_panels / 3), sharex=True, sharey=True)
        if n_panels == 1:
            axs = [axs]

        sho_params = ["log_sigma", "log_rho", "Q"]
        period_label = "log_period" if "log_period" in s else "period"
        rot_params = [period_label, "log_sigma_rot", "log_Q0", "log_dQ", "f"]
        param_text = {}
        for label in predictions:
            if label == "SHOTerm":
                names = sho_params
            elif label == "RotationTerm":
                names = rot_params
            else:
                names = []
            lines = []
            for n in names:
                if n in s:
                    lines.append(f"{n} = {float(s[n]):.4f}")
                elif n in (self._prior_params or {}) and self._prior_params[n]["type"] == "fixed":
                    lines.append(f"{n} = {self._prior_params[n]['value']:.4f} (fixed)")
            param_text[label] = ", \t\t".join(lines)

        for ax, (label, (mu, sigma)) in zip(axs, predictions.items()):
            ax.errorbar(
                self.x, self.y, yerr=self.yerr,
                fmt=".k", capsize=0, ms=3, alpha=0.4, zorder=1,
            )
            c = colors[label]
            ax.plot(xgrid, mu, color=c, lw=1.5, label=label, zorder=3)
            ax.fill_between(
                xgrid, mu - sigma, mu + sigma,
                color=c, alpha=0.25, zorder=2,
            )
            if param_text.get(label):
                ax.text(
                    0.01, 0.97, param_text[label],
                    transform=ax.transAxes, fontsize=18,
                    verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            ax.legend(fontsize=14, loc="upper right")
            ax.set_ylabel("Relative Flux", fontsize=22)
            ax.minorticks_on()

        axs[0].set_title(
            f"TIC {self.tic_id} — Sector {self.sector_label}  |  "
            f"MAP kernel components" + (f"  (Prot = {prot_map:.3f} d)" if prot_map is not None else ""),
            fontsize=20,
        )
        axs[-1].set_xlabel("Time [days]", fontsize=22)

        yspan = np.nanstd(self.y)
        for ax in axs:
            ax.set_ylim(self.y.mean() - 3.5 * yspan, self.y.mean() + 3.5 * yspan)
            ax.set_xlim(self.x.min(), self.x.max())

        fig.tight_layout()
        fname = os.path.join(
            output_dir,
            f"TIC{self.tic_id}_sector{self.sector_label}_kernel_components.png",
        )
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved kernel-component plot → {fname}\n")

    def _plot_results(self):
        """Generate GP fit and corner plots after fitting."""
        xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
        mu, var = self.predict(xgrid=xgrid)

        plot_dir = os.path.join(self.output_dir, f"plots/TIC{self.tic_id}")
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_gp_fit(xgrid, mu, var, plot_dir)
        self.plot_corner(plot_dir, show_priors=True)

    def plot_priors(self, return_fig=False, output_dir=None):
        """Plot the prior PDF for each free random variable."""
        priors = self._get_priors()
        names = [n for n in self.free_var_names if n in priors]
        n = len(names)

        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()

        for ax, name in zip(axes, names):
            dist = priors[name]
            try:
                lo = dist.ppf(0.001)
                hi = dist.ppf(0.999)
            except Exception:
                loc = getattr(dist, "mean", lambda: 0)()
                scale = getattr(dist, "std", lambda: 1)()
                lo, hi = loc - 4 * scale, loc + 4 * scale

            x = np.linspace(lo, hi, 500)
            pdf = dist.pdf(x)

            ax.plot(x, pdf, color="C0", lw=2)
            ax.fill_between(x, pdf, alpha=0.15, color="C0")
            ax.set_xlabel(name, fontsize=16)

            if self.map_soln is not None and name in self.map_soln:
                map_val = float(self.map_soln[name])
                ax.axvline(map_val, color="C1", lw=2, ls="--",
                           label=f"MAP = {map_val:.4g}")
                ax.legend(fontsize=9, loc="upper right")

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"TIC {self.tic_id} — Sector {self.sector_label} | Prior distributions",
            fontsize=14, y=1.01,
        )
        fig.tight_layout()

        output_dir = os.path.join(self.output_dir, f"prior_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_priors.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved prior plot → {fname}\n")
        if return_fig:
            return fig
        plt.close(fig)
        return None

    def plot_corner(self, output_dir=None, show_priors=True):
        """Create a corner plot of the posterior samples with MAP values marked."""
        free = self.free_var_names

        map_values = None
        if self.map_soln is not None:
            map_values = []
            for name in free:
                if name in self.map_soln:
                    map_values.append(float(self.map_soln[name]))
                else:
                    map_values.append(None)

        fig = corner.corner(
            self.trace,
            var_names=free,
            color="k",
            truths=map_values,
            truth_color="b",
            title_kwargs={"fontsize": 14},
            label_kwargs={"fontsize": 14},
            figsize=(20, 20),
        )

        if show_priors:
            priors = self._get_priors()
            ndim = len(free)
            axes = np.array(fig.axes).reshape((ndim, ndim))

            for i, name in enumerate(free):
                if name not in priors:
                    continue
                ax = axes[i, i]
                xlim = ax.get_xlim()
                x = np.linspace(xlim[0], xlim[1], 500)
                prior_pdf = priors[name].pdf(x)
                ylim = ax.get_ylim()
                peak = prior_pdf.max()
                if peak > 0:
                    prior_pdf = prior_pdf / peak * ylim[1] * 0.9
                ax.plot(x, prior_pdf, color="C0", lw=1.5, ls="--",
                        label="Prior" if i == 0 else None)

            axes[0, 0].legend(fontsize=10, loc="upper right")

        fig.suptitle(f"TIC {self.tic_id} — Sector {self.sector_label}", fontsize=20, y=1.02)

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"corner_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_corner.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved corner plot → {fname}\n")
        plt.close(fig)

    def plot_summary(self, output_dir=None):
        """Create a 3-panel summary plot: GP fit, phase-folded, Prot posterior."""
        posterior = self.trace.posterior
        if "period" in posterior:
            prot_samples = posterior["period"].values.flatten()
        elif "log_period" in posterior:
            prot_samples = np.exp(posterior["log_period"].values.flatten())
        else:
            raise KeyError("Neither 'period' nor 'log_period' found in posterior")
        prot_mean = float(np.mean(prot_samples))
        prot_std = float(np.std(prot_samples))

        xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
        mu, var = self.predict(xgrid=xgrid)
        sigma = np.sqrt(var)
        mu_map, var_map = self.predict_map(xgrid=xgrid)
        sigma_map = np.sqrt(var_map) if var_map is not None else None

        prot_catalog = None
        try:
            params = get_object_params(self.tic_id, self.catalog_path)
            prot_catalog = params.get("Prot")
            if prot_catalog is not None and np.isnan(prot_catalog):
                prot_catalog = None
        except Exception:
            pass

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))
        plt.subplots_adjust(hspace=0.25)

        axs[0].scatter(self.x, self.y, color="k", s=3, label="Rotation data")
        if self.x_transit is not None and self.y_transit is not None:
            axs[0].scatter(self.x_transit, self.y_transit, color="C1", s=3, alpha=0.6, label="Transit points")
        if mu_map is not None:
            axs[0].plot(xgrid, mu_map, color="b", lw=1.5, alpha=0.7, label="MAP prediction")
        if sigma_map is not None:
            axs[0].fill_between(xgrid, mu_map - sigma_map, mu_map + sigma_map, color="b", alpha=0.2)
        axs[0].set_xlim(self.x.min(), self.x.max())
        axs[0].set_xlabel("Time [days]", fontsize=16)
        axs[0].set_ylabel("Relative Flux", fontsize=16)
        axs[0].set_title(f"TIC {self.tic_id} — Sector {self.sector_label}", fontsize=18)
        axs[0].legend(fontsize=12)
        axs[0].minorticks_on()

        axs[1].scatter(self.x % prot_mean, self.y, color="k", s=3,
                       label=f"Prot = {prot_mean:.3f} d")
        if self.x_transit is not None and self.y_transit is not None:
            axs[1].scatter(self.x_transit % prot_mean, self.y_transit, color="C1", s=3, alpha=0.6)
        axs[1].set_xlim(0, prot_mean)
        axs[1].set_xlabel("Phase [days]", fontsize=16)
        axs[1].set_ylabel("Relative Flux", fontsize=16)
        axs[1].legend(fontsize=12)
        axs[1].minorticks_on()

        axs[2].hist(prot_samples, bins="fd", histtype="step", color="k", density=True)
        axs[2].axvline(prot_mean, color="r", label=f"GP Prot = {prot_mean:.3f} +/- {prot_std:.3f}")
        axs[2].axvspan(prot_mean - prot_std, prot_mean + prot_std, color="r", alpha=0.2)
        if prot_catalog is not None:
            axs[2].axvline(prot_catalog, color="b", label=f"Catalog Prot = {prot_catalog:.3f}")
        axs[2].set_xlabel("Rotation Period [days]", fontsize=16)
        axs[2].set_ylabel("Posterior Density", fontsize=16)
        axs[2].legend(fontsize=12)
        axs[2].minorticks_on()

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"summary_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_gp_summary.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved GP summary plot → {fname}\n")
        plt.close(fig)

    def plot_psd(self, output_dir=None, return_fig=False, show_kernel=True,
                 freq_min=None, freq_max=None, n_freq=5000, soln=None,
                 log_spacing=False):
        """Plot the power spectral density of the lightcurve."""
        fig, ax = plt.subplots(figsize=(14, 6))
        self._draw_psd(ax, soln=soln, show_kernel=show_kernel,
                       freq_min=freq_min, freq_max=freq_max, n_freq=n_freq,
                       log_spacing=log_spacing)
        fig.tight_layout()

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"psd_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_psd.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved PSD plot → {fname}\n")
        if return_fig:
            return fig, ax
        plt.close(fig)

    def plot_acf(self, output_dir=None, return_fig=False, max_lag=None,
                 n_lags=500, show_kernel=True, soln=None):
        """Plot the autocorrelation function of the data and the GP kernel."""
        fig, ax = plt.subplots(figsize=(14, 6))
        self._draw_acf(ax, soln=soln, show_kernel=show_kernel,
                       max_lag=max_lag, n_lags=n_lags)
        fig.tight_layout()

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"acf_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_acf.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved ACF plot → {fname}\n")
        if return_fig:
            return fig, ax
        plt.close(fig)

    def plot_psd_acf(self, output_dir=None, return_fig=False, soln=None,
                     show_kernel=True, freq_min=None, freq_max=None,
                     n_freq=5000, log_spacing=False, max_lag=None, n_lags=500,
                     plot_suffix=""):
        """Two-panel plot combining PSD (top) and ACF (bottom)."""
        fig, (ax_psd, ax_acf) = plt.subplots(2, 1, figsize=(14, 12))

        self._draw_psd(ax_psd, soln=soln, show_kernel=show_kernel,
                       freq_min=freq_min, freq_max=freq_max, n_freq=n_freq,
                       log_spacing=log_spacing)
        self._draw_acf(ax_acf, soln=soln, show_kernel=show_kernel,
                       max_lag=max_lag, n_lags=n_lags)

        fig.tight_layout()

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"psd_acf_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"TIC{self.tic_id}_sector{self.sector_label}_psd_acf{plot_suffix}.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved PSD+ACF plot → {fname}\n")
        if return_fig:
            return fig, (ax_psd, ax_acf)
        plt.close(fig)

    def _plot_phase_fold_common(self, period, n_bins, output_dir, return_fig,
                                soln, show_gp, show_transit):
        """Shared phase-fold plotting logic. Returns (fig, ax, phase, period, output_dir).

        Subclasses should call this, then optionally overlay GP prediction.
        """
        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        if soln is None:
            soln = self.map_soln

        if period is None:
            if soln is not None:
                if "period" in soln:
                    period = float(soln["period"])
                elif "log_period" in soln:
                    period = float(np.exp(soln["log_period"]))
            if period is None:
                period = self.prot_init
            if period is None:
                raise ValueError("No period specified and none found in solution or catalog.")

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"phase_fold_plots/TIC{self.tic_id}")
        os.makedirs(output_dir, exist_ok=True)

        phase = (self.x % period) / period

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.full(n_bins, np.nan)
        bin_errs = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if mask.sum() > 0:
                w = 1.0 / self.yerr[mask] ** 2
                bin_means[i] = np.average(self.y[mask], weights=w)
                bin_errs[i] = 1.0 / np.sqrt(np.sum(w))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.scatter(phase, self.y, c="k", s=3, alpha=0.3, zorder=1, label="Data")

        if show_transit and self.x_transit is not None and self.y_transit is not None:
            phase_transit = (self.x_transit % period) / period
            ax.scatter(phase_transit, self.y_transit, c="C1", s=3,
                       alpha=0.4, zorder=1, label="Transit points")

        valid = np.isfinite(bin_means)
        ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid],
                    fmt="o", color="C3", ms=5, capsize=2, zorder=3,
                    label=f"Binned mean ({n_bins} bins)")

        return fig, ax, soln, period, output_dir, return_fig

    def _finalize_phase_fold(self, fig, ax, period, output_dir, return_fig):
        """Finish and save phase-fold plot."""
        ax.set_xlim(0, 1)
        ax.set_xlabel("Phase", fontsize=16)
        ax.set_ylabel("Relative Flux", fontsize=16)
        ax.set_title(f"TIC {self.tic_id} — Sector {self.sector_label}  |  "
                      f"Phase-folded (P = {period:.4f} d)", fontsize=18)
        ax.legend(fontsize=14)
        ax.minorticks_on()
        fig.tight_layout()

        fname = os.path.join(output_dir,
                             f"TIC{self.tic_id}_sector{self.sector_label}_phase_fold.png")
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved phase-fold plot → {fname}\n")
        if return_fig:
            return fig, ax
        plt.close(fig)

    def _detect_kernel_flags(self):
        """Set fit_shoterm/fit_rotationterm from _prior_params."""
        p = self._prior_params
        rv_names = set(p.keys())

        self.fit_shoterm = (
            ("sigma" in rv_names or "log_sigma" in rv_names) and
            ("rho" in rv_names or "log_rho" in rv_names) and
            ("Q" in rv_names or "log_Q" in rv_names)
        )
        self.fit_rotationterm = (
            ("period" in rv_names or "log_period" in rv_names) and
            ("sigma_rot" in rv_names or "log_sigma_rot" in rv_names) and
            ("Q0" in rv_names or "log_Q0" in rv_names) and
            ("dQ" in rv_names or "log_dQ" in rv_names) and
            ("f" in rv_names or "log_f" in rv_names)
        )

        if not self.fit_shoterm and not self.fit_rotationterm:
            raise ValueError(
                "No kernel terms enabled — at least one of SHOTerm or "
                "RotationTerm must be specified in prior_params."
            )
