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


class WalkerPlotCallback:
    """Live trace plot updated during PyMC sampling.

    Parameters
    ----------
    var_names : list of str
        Model variable names to display.
    save_path : str
        File path to save the plot. Overwritten on every update.
    update_every : int
        Refresh the plot every N draws.
    """

    def __init__(self, var_names, save_path, update_every=50, ylims=None, priors=None,
                 map_soln=None, prior_params=None):
        self.var_names = var_names
        self.save_path = save_path
        self.update_every = update_every
        self._chain_data = {}  # {chain_idx: {var_name: array}}
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

            # --- Right column: rotated prior PDF (shares y-axis with trace) ---
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
            
            # Text to the right: prior label + MAP value
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
        # trace is the per-chain NDArray backend; draw.chain identifies which chain
        if draw.draw_idx == 0 or draw.draw_idx % self.update_every != 0:
            return

        # Pull all draws so far for this chain
        chain = draw.chain
        chain_vals = {}
        for name in self.var_names:
            try:
                vals = trace.get_values(name)[:draw.draw_idx + 1]  # slice to draws completed so far
                chain_vals[name] = np.atleast_1d(vals).reshape(len(vals), -1)
            except Exception:
                return
        self._chain_data[chain] = chain_vals

        # Redraw all accumulated chains
        for i, name in enumerate(self.var_names):

            # --- Left column: trace ---
            self.axes[i, 0].cla()
            for c in sorted(self._chain_data):
                if name in self._chain_data[c]:
                    self.axes[i, 0].plot(self._chain_data[c][name], lw=1)
            
            self.axes[i, 0].set_ylabel(name, fontsize=25)
        self.axes[-1, 0].set_xlabel("draw", fontsize=25)
            
        self.fig.savefig(self.save_path, bbox_inches="tight")
        plt.pause(0.001)


class GPFit:
    """GP rotation-period fitter for a single TESS target.

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
        """Sampled (non-fixed) variable names, taken directly from _prior_params keys."""
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

        # Per-sector state (populated by load_data / fit / reload)
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
        """Load rotation-masked lightcurve for one or more sectors.

        Parameters
        ----------
        sector : int or list of int
            Sector number(s). When a list is given the light curves are
            concatenated in time order and ``self.sector`` is stored as
            the list.

        Returns
        -------
        x, y, yerr : np.ndarray
        """
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

        # sort by time
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
            
        # downsample if requested (after binning, to speed up initial testing)
        if self.downsample > 1:
            x = x[::self.downsample]
            y = y[::self.downsample]
            yerr = yerr[::self.downsample]

        # normalize flux to zero mean and scale by stddev (after binning)
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
    # Model building
    # ------------------------------------------------------------------

    def build_pymc_model(self, prior_params=None):
        """Build a PyMC GP model with SHOTerm + RotationTerm kernel."""
        import pymc as pm
        import pytensor.tensor as pt
        from celerite2.pymc import GaussianProcess, terms

        x, y, yerr = self.x, self.y, self.yerr

        if prior_params is not None:
            self._prior_params = prior_params
        p = self._prior_params

        with pm.Model() as model:
            rv = {}
            for name, spec in p.items():
                if spec["type"] == "normal":
                    rv[name] = pm.Normal(name, mu=spec["mu"], sigma=spec["sigma"])
                elif spec["type"] == "half_normal":
                    rv[name] = pm.HalfNormal(name, sigma=spec["sigma"])
                elif spec["type"] == "log_normal":
                    rv[name] = pm.LogNormal(name, mu=spec["mu"], sigma=spec["sigma"])
                elif spec["type"] == "truncated_normal":
                    rv[name] = pm.TruncatedNormal(name, mu=spec["mu"], sigma=spec["sigma"],
                                                  lower=spec["lower"], upper=spec["upper"])
                elif spec["type"] == "uniform":
                    rv[name] = pm.Uniform(name, lower=spec["lower"], upper=spec["upper"])
                elif spec["type"] == "inverse_gamma":
                    ig = estimate_inverse_gamma_parameters(spec["lower"], spec["upper"])
                    rv[name] = pm.InverseGamma(name, alpha=ig["alpha"], beta=ig["beta"])
                elif spec["type"] == "fixed":
                    rv[name] = spec["value"]
                else:
                    raise ValueError(f"Unknown prior type {spec['type']!r} for parameter {name!r}")

            def resolve(canonical):
                """Return the pytensor expression for `canonical`.

                Checks for ``log_{canonical}`` first; if present, returns
                ``pt.exp`` of that RV (log-parameterisation).  Falls back to
                ``canonical`` directly (linear parameterisation).
                """
                log_name = f"log_{canonical}"
                if log_name in rv:
                    return pt.exp(rv[log_name])
                elif canonical in rv:
                    return rv[canonical]
                raise KeyError(
                    f"Neither {log_name!r} nor {canonical!r} found in prior_params"
                )
                
            kernel = None
            if ("sigma" in rv or "log_sigma" in rv) and ("rho" in rv or "log_rho" in rv) and ("Q" in rv or "log_Q" in rv):
                self.fit_shoterm = True
                kernel = terms.SHOTerm(
                    sigma=resolve("sigma"),
                    rho=resolve("rho"),
                    Q=resolve("Q"),
                )
            else:
                self.fit_shoterm = False

            if ("period" in rv or "log_period" in rv) and ("sigma_rot" in rv or "log_sigma_rot" in rv) and ("Q0" in rv or "log_Q0" in rv) and ("dQ" in rv or "log_dQ" in rv) and ("f" in rv or "log_f" in rv):
                self.fit_rotationterm = True
                rot_term = terms.RotationTerm(
                    period=resolve("period"),
                    sigma=resolve("sigma_rot"),
                    Q0=resolve("Q0"),
                    dQ=resolve("dQ"),
                    f=resolve("f"),
                )
                kernel = kernel + rot_term if kernel is not None else rot_term
            else:
                self.fit_rotationterm = False

            if kernel is None:
                raise ValueError("No kernel terms enabled — at least one of SHOTerm or RotationTerm must be specified in prior_params.")

            jitter = resolve("jitter")
            gp = GaussianProcess(
                kernel,
                t=x,
                diag=yerr**2 + jitter**2,
                mean=resolve("mean"),
                quiet=True,
            )
            gp.marginal("gp", observed=y)

        return model

    def build_pymc3_model(self):
        """Build a PyMC GP model for use with pymc_ext MAP optimization."""
        return NotImplementedError

    # ------------------------------------------------------------------
    # MAP optimization
    # ------------------------------------------------------------------

    def find_map(self, sector, sampler="pymc", start=None):
        """Find the MAP solution for a sector.

        Loads data (if not already loaded for this sector), builds the model,
        and runs MAP optimization. Stores the result in ``self.map_soln``.

        Parameters
        ----------
        sector : int
            Sector number.
        sampler : str
            Backend to use for the model: "pymc", "pymc3", or "numpyro".

        Returns
        -------
        map_soln : dict
            The MAP parameter values.
        """
        if self.sector != sector or self.x is None:
            self.load_data(sector)
            
        if not hasattr(self, "_prior_params") or self._prior_params is None:
            raise ValueError("Prior parameters must be set before finding MAP. Call `set_prior_params` first.")

        if sampler == "pymc":
            model = self.build_pymc_model()
            self._model = model
            self.map_soln = self._find_map_pymc(model, start=start)
        else:
            return NotImplementedError("MAP optimization with pymc3 is not yet implemented.")

        return self.map_soln

    def _find_map_pymc(self, model, start=None):
        """Find MAP using pm.find_MAP (three-stage)."""
        import pymc as pm

        free = set(self.free_var_names)

        # Determine which period parameterisation is in use
        use_log_period = "log_period" in free

        def get_vars(names):
            """Return model RVs for the given names, skipping any that are fixed."""
            return [model.named_vars[n] for n in names if n in free and n in model.named_vars]

        def filter_point(point):
            """Keep only keys matching build_pymc_model's variable names (prior_params keys).

            pm.find_MAP returns both the untransformed keys (e.g. "period") and
            PyMC-internal transformed keys (e.g. "period_interval__").  When both
            are present in a subsequent start dict, PyMC treats the transformed key
            as authoritative, so any clamping applied to the untransformed key is
            silently ignored.  Filtering to just the prior_params keys ensures that
            only the untransformed values are forwarded, which PyMC then re-transforms
            correctly for the next optimisation stage.
            """
            return {k: v for k, v in point.items() if k in free}

        def clamp_bounded(point):
            """Clamp bounded parameters strictly inside their bounds.

            Values exactly at a boundary produce ±inf in PyMC's interval
            transform, crashing subsequent find_MAP or sampling calls.
            """
            if not self._prior_params:
                return point
            eps = 1e-3
            for name, spec in self._prior_params.items():
                if spec["type"] in ("uniform", "truncated_normal") and name in point:
                    span = spec["upper"] - spec["lower"]
                    lo = spec["lower"] + eps * span
                    hi = spec["upper"] - eps * span
                    point[name] = float(np.clip(float(point[name]), lo, hi))
            return point

        # Stage 1: initialise mean + period (only if they are free)
        period_key = "log_period" if use_log_period else "period"
        stage1 = [n for n in ["mean", period_key] if n in free]
        
        if start is None:
            start = {}
            if "mean" in free:
                start["mean"] = float(np.mean(self.y))
            if use_log_period and "log_period" in free:
                start["log_period"] = float(np.log(self.prot_init))
            elif "period" in free:
                start["period"] = self.prot_init
        else:
            # Filter start to only include keys that are free model variables,
            # otherwise PyMC raises KeyError for unknown variable names.
            start = {k: v for k, v in start.items() if k in free}

        # Stage 2: kernel / noise params (everything except mean & period)
        stage2 = [n for n in self.free_var_names if n not in {"mean", period_key}]

        with model:
            if stage1:
                map_soln = clamp_bounded(filter_point(pm.find_MAP(start=start, vars=get_vars(stage1))))
            else:
                map_soln = {}

            if stage2:
                map_soln = clamp_bounded(filter_point(pm.find_MAP(start=map_soln, vars=get_vars(stage2))))

            # Stage 3: all free vars jointly
            map_soln = clamp_bounded(filter_point(pm.find_MAP(start=map_soln, vars=get_vars(self.free_var_names))))

        return map_soln

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_pymc(self, model, tune, draws, chains, cores, target_accept,
                     map_soln=None, callback=None, step_scale=None, init_jitter=0.1):
        """Run MCMC sampling with the PyMC NUTS sampler."""
        import pymc as pm
        import pandas as pd

        if map_soln is None:
            map_soln = self._find_map_pymc(model)
        elif isinstance(map_soln, pd.Series):
            map_soln = {k: float(map_soln[k]) for k in self.free_var_names if k in map_soln.index}

        # Clamp all uniform-prior parameters strictly inside their bounds.
        # Values exactly at a boundary produce ±inf in PyMC's interval transform,
        # which causes NaN logp and a SamplingError during NUTS initialisation.
        if isinstance(map_soln, dict) and self._prior_params:
            eps = 1e-3
            for name, spec in self._prior_params.items():
                if spec["type"] in ("uniform", "truncated_normal") and name in map_soln:
                    span = spec["upper"] - spec["lower"]
                    lo = spec["lower"] + eps * span
                    hi = spec["upper"] - eps * span
                    map_soln[name] = float(np.clip(map_soln[name], lo, hi))

        # Create per-chain jittered starting points so walkers don't all
        # begin at the identical MAP solution.
        if isinstance(map_soln, dict) and map_soln:
            rng = np.random.default_rng()
            initvals = []
            for _ in range(chains):
                jittered = {}
                for k, v in map_soln.items():
                    scale = init_jitter * abs(v) if v != 0 else init_jitter
                    jittered[k] = v + rng.normal(0, scale)
                # Re-clamp jittered values inside uniform/truncated_normal bounds
                if self._prior_params:
                    for name, spec in self._prior_params.items():
                        if spec["type"] in ("uniform", "truncated_normal") and name in jittered:
                            eps = 1e-3
                            span = spec["upper"] - spec["lower"]
                            lo = spec["lower"] + eps * span
                            hi = spec["upper"] - eps * span
                            jittered[name] = float(np.clip(jittered[name], lo, hi))
                initvals.append(jittered)
        else:
            initvals = map_soln

        with model:
            sample_kwargs = dict(
                tune=tune,
                draws=draws,
                chains=chains,
                cores=cores,
                init="adapt_diag",
                initvals=initvals,
                target_accept=target_accept,
                return_inferencedata=True,
                callback=callback,
            )
            if step_scale is not None:
                sample_kwargs["step_scale"] = step_scale

            trace = pm.sample(**sample_kwargs)

        return trace

    # ------------------------------------------------------------------
    # Fitting (main entry point)
    # ------------------------------------------------------------------

    def fit(self, x=None, y=None, yerr=None, sampler="pymc", tune=2000, draws=1000, chains=4,
            target_accept=0.9, cores=1, chain_method="parallel", map_soln=None,
            callback=None, step_scale=None, init_jitter=0.1):
        """Fit a GP model to a single sector's rotation-masked lightcurve.

        Parameters
        ----------
        sector : int
            Sector number.
        sampler : str
            Sampler backend: "pymc", "pymc3", or "numpyro".
        tune, draws, chains : int
            MCMC sampling parameters.
        target_accept : float
            Target acceptance rate for NUTS.
        cores : int
            Number of cores for sampling (PyMC only).
        chain_method : str
            NumPyro chain method: "sequential", "parallel", or "vectorized".
        step_scale : float, optional
            Initial step size for NUTS before adaptation. Larger values
            (e.g. 0.5–1.0) start with bigger steps during tuning. If None,
            the sampler default is used (PyMC ~0.25, NumPyro ~0.1).

        Returns
        -------
        trace : arviz.InferenceData
            The posterior samples.
        """
        # Load data if not already loaded for this sector
        if (x is None) or (y is None) or (yerr is None):
            x, y, yerr = self.load_data(self.sector)
            
        if map_soln is None:
            map_soln = {} 

        if sampler == "pymc":
            # Reuse model from find_map if available, otherwise build new
            model = getattr(self, "_model", None) or self.build_pymc_model()
            self.trace = self._sample_pymc(
                model, tune=tune, draws=draws, chains=chains,
                cores=cores, target_accept=target_accept,
                map_soln=map_soln, callback=callback,
                step_scale=step_scale, init_jitter=init_jitter
            )
        else:
            raise NotImplementedError(f"Sampler {sampler!r} is not yet implemented. Use 'sampler=\"pymc\"'.")

        self._save_results()
        self._plot_results()

        return self.trace

    # ------------------------------------------------------------------
    # Post-hoc GP prediction
    # ------------------------------------------------------------------

    def _build_kernel(self, resolve, terms_module, which="full"):
        """Build a celerite2 kernel using a ``resolve(canonical)`` callable.

        Parameters
        ----------
        resolve : callable
            ``resolve(name)`` returns the *physical* (non-log) value for
            *name*.  Works with both MAP-dict resolvers and trace-sample
            resolvers.
        terms_module : module
            ``celerite2.terms`` (or its ``.pymc`` / ``.jax`` variant).
        which : str
            ``"full"`` (default), ``"sho"``, or ``"rot"`` — which component(s)
            to include.

        Returns
        -------
        kernel
            A celerite2 kernel term (or sum of terms).
        """
        kernel = None

        if which in ("full", "sho") and getattr(self, "fit_shoterm", True):
            kernel = terms_module.SHOTerm(
                sigma=resolve("sigma"),
                rho=resolve("rho"),
                Q=resolve("Q"),
            )

        if which in ("full", "rot") and getattr(self, "fit_rotationterm", True):
            rot = terms_module.RotationTerm(
                sigma=resolve("sigma_rot"),
                period=resolve("period"),
                Q0=resolve("Q0"),
                dQ=resolve("dQ"),
                f=resolve("f"),
            )
            kernel = kernel + rot if kernel is not None else rot

        if kernel is None:
            raise RuntimeError("No kernel components are enabled.")
        return kernel

    def predict(self, xgrid=None):
        """Compute GP mean prediction from posterior samples.

        Parameters
        ----------
        xgrid : np.ndarray, optional
            Fine grid of times at which to predict. If None, predicts at self.x.

        Returns
        -------
        mu, var : np.ndarray
            Mean and variance of the GP prediction across posterior samples.
        """
        from celerite2 import GaussianProcess as celeriteGP
        from celerite2 import terms as celerite_terms

        if xgrid is None:
            xgrid = self.x

        posterior = self.trace.posterior
        flat = posterior.stack(sample=("chain", "draw"))

        n_samples = flat.sizes["sample"]
        preds = np.empty((n_samples, len(xgrid)))

        def resolve(canonical, idx):
            """Return physical (non-log) value for `canonical` from sample `idx`.

            Checks ``log_{canonical}`` first (→ exp), then ``canonical``
            directly, then falls back to a fixed value in _prior_params.
            """
            log_name = f"log_{canonical}"
            if log_name in flat:
                return float(np.exp(flat[log_name].isel(sample=idx)))
            if canonical in flat:
                return float(flat[canonical].isel(sample=idx))
            p = self._prior_params or {}
            if log_name in p and p[log_name]["type"] == "fixed":
                return float(np.exp(p[log_name]["value"]))
            if canonical in p and p[canonical]["type"] == "fixed":
                return float(p[canonical]["value"])
            raise KeyError(f"Parameter {canonical!r} not found in trace or _prior_params")

        for i in range(n_samples):
            kernel = self._build_kernel(lambda c: resolve(c, i), celerite_terms)
            jitter = resolve("jitter", i)
            gp = celeriteGP(kernel, mean=resolve("mean", i))
            gp.compute(self.x, diag=self.yerr**2 + jitter**2)
            preds[i] = gp.predict(self.y, t=xgrid)

        return preds.mean(axis=0), preds.var(axis=0)

    def predict_map(self, xgrid=None):
        """Compute GP prediction using the MAP solution.

        Parameters
        ----------
        xgrid : np.ndarray, optional
            Times at which to predict. If None, predicts at self.x.

        Returns
        -------
        mu_map : np.ndarray
            GP mean prediction at the MAP solution.
        """
        from celerite2 import GaussianProcess as celeriteGP
        from celerite2 import terms as celerite_terms

        if self.map_soln is None:
            return None

        if xgrid is None:
            xgrid = self.x

        s = self.map_soln

        def resolve(canonical):
            """Return physical (non-log) value for `canonical` from the MAP solution.

            Checks ``log_{canonical}`` first (→ exp), then ``canonical``
            directly, then falls back to a fixed value in _prior_params.
            """
            log_name = f"log_{canonical}"
            if log_name in s:
                return float(np.exp(s[log_name]))
            if canonical in s:
                return float(s[canonical])
            p = self._prior_params or {}
            if log_name in p and p[log_name]["type"] == "fixed":
                return float(np.exp(p[log_name]["value"]))
            if canonical in p and p[canonical]["type"] == "fixed":
                return float(p[canonical]["value"])
            raise KeyError(f"Parameter {canonical!r} not found in map_soln or _prior_params")

        kernel = self._build_kernel(resolve, celerite_terms)
        jitter = resolve("jitter")
        gp = celeriteGP(kernel, mean=resolve("mean"))
        gp.compute(self.x, diag=self.yerr**2 + jitter**2)

        return gp.predict(self.y, t=xgrid, return_var=True)

    def plot_kernel_components(self, output_dir=None, soln=None):
        """Plot lightcurve with individual GP kernel components.

        Shows three GP predictions overlaid on the data:
        - SHOTerm only (non-periodic / granulation)
        - RotationTerm only (quasi-periodic stellar rotation)
        - Full kernel (SHOTerm + RotationTerm)

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the plot. Defaults to
            ``self.output_dir/kernel_component_plots/TIC{tic_id}``.
        soln : dict, optional
            Parameter solution dict to use. If None, uses ``self.map_soln``.
        """
        from celerite2 import GaussianProcess as celeriteGP
        from celerite2 import terms as celerite_terms

        if soln is None:
            soln = self.map_soln
        if soln is None:
            raise RuntimeError("No solution available. Pass soln or run find_map() first.")

        if output_dir is None:
            output_dir = os.path.join(
                self.output_dir, f"kernel_component_plots/TIC{self.tic_id}"
            )
        os.makedirs(output_dir, exist_ok=True)

        s = soln

        def resolve(canonical):
            log_name = f"log_{canonical}"
            if log_name in s:
                return float(np.exp(s[log_name]))
            if canonical in s:
                return float(s[canonical])
            p = self._prior_params or {}
            if log_name in p and p[log_name]["type"] == "fixed":
                return float(np.exp(p[log_name]["value"]))
            if canonical in p and p[canonical]["type"] == "fixed":
                return float(p[canonical]["value"])
            raise KeyError(
                f"Parameter {canonical!r} not found in map_soln or _prior_params"
            )

        mean_val = resolve("mean")
        jitter = resolve("jitter")
        diag = self.yerr**2 + jitter**2
        xgrid = np.linspace(self.x.min(), self.x.max(), 2000)

        # --- Build individual kernels and predict ---
        predictions = {}
        colors = {}

        if getattr(self, "fit_shoterm", False):
            sho_kernel = self._build_kernel(resolve, celerite_terms, which="sho")
            gp = celeriteGP(sho_kernel, mean=mean_val)
            gp.compute(self.x, diag=diag)
            mu, var = gp.predict(self.y, t=xgrid, return_var=True)
            predictions["SHOTerm"] = (mu, np.sqrt(var))
            colors["SHOTerm"] = "C2"

        if getattr(self, "fit_rotationterm", False):
            rot_kernel = self._build_kernel(resolve, celerite_terms, which="rot")
            gp = celeriteGP(rot_kernel, mean=mean_val)
            gp.compute(self.x, diag=diag)
            mu, var = gp.predict(self.y, t=xgrid, return_var=True)
            predictions["RotationTerm"] = (mu, np.sqrt(var))
            colors["RotationTerm"] = "C0"

        if getattr(self, "fit_shoterm", False) and getattr(self, "fit_rotationterm", False):
            full_kernel = self._build_kernel(resolve, celerite_terms, which="full")
            gp = celeriteGP(full_kernel, mean=mean_val)
            gp.compute(self.x, diag=diag)
            mu, var = gp.predict(self.y, t=xgrid, return_var=True)
            predictions["SHOTerm + RotationTerm"] = (mu, np.sqrt(var))
            colors["SHOTerm + RotationTerm"] = "C3"

        try:
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

        # Build parameter text for each component
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
    # Plotting
    # ------------------------------------------------------------------

    def plot_raw(self, sector=None, output_dir=None):
        """Plot the raw lightcurve for a sector showing rotation and transit data.

        Reads directly from the masked HDF5 file and plots both the
        rotation-masked and transit-masked points in their original flux
        units (before normalization/binning).

        Parameters
        ----------
        sector : int, optional
            Sector number. Defaults to ``self.sector``.
        output_dir : str, optional
            Directory to save the plot. Defaults to ``self.output_dir/plots``.
        """
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
        """Plot the lightcurve with the MAP GP prediction overlaid.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the plot. Defaults to ``self.output_dir/plots``.
        return_fig : bool, optional
            If True, return (fig, axs) without closing the figure instead of
            closing it. Useful for callers that want to annotate the figure
            further before saving. Default is False.
        """
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
        
    def _plot_results(self):
        """Generate GP fit and corner plots after fitting."""
        xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
        mu, var = self.predict(xgrid=xgrid)

        plot_dir = os.path.join(self.output_dir, f"plots/TIC{self.tic_id}")
        os.makedirs(plot_dir, exist_ok=True)

        self.plot_gp_fit(xgrid, mu, var, plot_dir)
        self.plot_corner(plot_dir, show_priors=True)

    def _get_priors(self):
        """Return scipy prior distributions for each parameter.

        Reads from ``self._prior_params`` (populated by ``build_pymc_model``)
        so that the scipy distributions always match the PyMC model exactly.
        If ``_prior_params`` has not been set yet, it is computed from the
        current data using the same defaults as ``build_pymc_model``.

        Returns
        -------
        dict
            Maps parameter name to a scipy.stats frozen distribution.
        """
        from scipy import stats

        if self._prior_params is None:
            # Compute defaults matching build_pymc_model so _get_priors can be
            # called before build_pymc_model (e.g. for plotting).
            self._prior_params = {
                "mean":          {"type": "normal",  "mu": float(np.mean(self.y)),            "sigma": float(np.std(self.y))},
                "log_jitter":    {"type": "normal",  "mu": float(np.log(np.mean(self.yerr))), "sigma": 2.0},
                "log_sigma":     {"type": "normal",  "mu": float(np.log(np.var(self.y))),     "sigma": 5.0},
                "log_rho":       {"type": "uniform", "lower": -10.0,                           "upper": 5.0},
                "period":        {"type": "uniform", "lower": 0.5 * self.prot_init,            "upper": 1.5 * self.prot_init},
                "log_sigma_rot": {"type": "normal",  "mu": float(np.log(np.var(self.y))),     "sigma": 5.0},
                "log_Q0":        {"type": "uniform", "lower": -10.0,                           "upper": 10.0},
                "log_dQ":        {"type": "normal",  "mu": 2.0,                                "sigma": 5.0},
                "f":             {"type": "uniform", "lower": 0.1,                             "upper": 1.0},
                "Q":             {"type": "fixed",   "value": 1/3},
            }

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
            # "fixed" params are intentionally omitted — no distribution to plot
        return priors

    def get_walker_ylims(self, sigma_factor=5):
        """Return y-axis limits for ``WalkerPlotCallback`` based on the priors.

        * uniform       → ``(lower, upper)``
        * normal        → ``(mu ± sigma_factor * sigma)``
        * inverse_gamma → ``(ppf(0.001), ppf(0.999))`` of the fitted distribution
        * fixed         → excluded

        Returns
        -------
        dict
            Maps parameter name to ``(lo, hi)`` tuples.
        """
        priors = self._get_priors()  # also populates _prior_params if needed

        ylims = {}
        for name, p in self._prior_params.items():
            if p["type"] == "uniform":
                ylims[name] = (p["lower"], p["upper"])
            elif p["type"] == "truncated_normal":
                ylims[name] = (p["lower"], p["upper"])
            elif p["type"] == "normal":
                ylims[name] = (p["mu"] - sigma_factor * p["sigma"], p["mu"] + sigma_factor * p["sigma"])
            elif name in priors:
                # Generic ppf-based limits for half_normal, log_normal, inverse_gamma, etc.
                dist = priors[name]
                ylims[name] = (float(dist.ppf(0.001)), float(dist.ppf(0.999)))
        return ylims

    def plot_priors(self, return_fig=False, output_dir=None):
        """Plot the prior PDF for each free random variable.

        Each parameter gets its own subplot. If ``map_soln`` is available,
        a vertical line is drawn at the MAP value with a legend label.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the figure. If None the figure is not saved.
        return_fig : bool
            If True, return the Figure instead of closing it.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        priors = self._get_priors()
        names = [n for n in self.free_var_names if n in priors]
        n = len(names)

        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()

        for ax, name in zip(axes, names):
            dist = priors[name]

            # Build x range: use ppf if available, else fall back to mean ± 4σ
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

        # Hide unused subplots
        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"TIC {self.tic_id} — Sector {self.sector_label} | Prior distributions",
            fontsize=14,
            y=1.01,
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
        """Create a corner plot of the posterior samples with MAP values marked.

        Parameters
        ----------
        output_dir : str
            Directory to save the plot.
        show_priors : bool
            If True, overlay the prior PDF on each 1-D diagonal histogram.
        """
        free = self.free_var_names

        # Extract MAP values in free_var_names order for the truths argument
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
                # Scale prior to match histogram height
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

        # Panel 1: Lightcurve + GP prediction
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

        # Panel 2: Phase-folded lightcurve
        axs[1].scatter(self.x % prot_mean, self.y, color="k", s=3,
                       label=f"Prot = {prot_mean:.3f} d")
        if self.x_transit is not None and self.y_transit is not None:
            axs[1].scatter(self.x_transit % prot_mean, self.y_transit, color="C1", s=3, alpha=0.6)
        axs[1].set_xlim(0, prot_mean)
        axs[1].set_xlabel("Phase [days]", fontsize=16)
        axs[1].set_ylabel("Relative Flux", fontsize=16)
        axs[1].legend(fontsize=12)
        axs[1].minorticks_on()

        # Panel 3: Prot posterior histogram
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

    def plot_phase_fold(self, period=None, n_bins=50, output_dir=None,
                        return_fig=False, soln=None, show_gp=False, show_transit=False):
        """Plot the phase-folded lightcurve at a given period.

        Parameters
        ----------
        period : float, optional
            Folding period in days.  If None, uses the period from ``soln``
            (or ``self.map_soln``).
        n_bins : int, optional
            Number of phase bins for the binned average.  Default is 50.
        output_dir : str, optional
            Directory to save the plot.  Defaults to
            ``self.output_dir/phase_fold_plots/TIC{tic_id}``.
        return_fig : bool, optional
            If True, return ``(fig, ax)`` instead of closing the figure.
        soln : dict, optional
            Parameter solution dict.  If None, uses ``self.map_soln``.
        show_gp : bool, optional
            If True and a solution with a GP prediction is available,
            overlay the phase-folded GP prediction.  Default is True.

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes or None
            Returned only when ``return_fig`` is True.
        """
        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        if soln is None:
            soln = self.map_soln

        # Determine period
        if period is None:
            if soln is not None:
                if "period" in soln:
                    period = float(soln["period"])
                elif "log_period" in soln:
                    period = float(np.exp(soln["log_period"]))
            if period is None:
                period = self.prot_init
            if period is None:
                raise ValueError(
                    "No period specified and none found in solution or catalog."
                )

        if output_dir is None:
            output_dir = os.path.join(
                self.output_dir, f"phase_fold_plots/TIC{self.tic_id}"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Phase-fold data
        phase = (self.x % period) / period
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        y_sorted = self.y[sort_idx]
        yerr_sorted = self.yerr[sort_idx]

        # Binned phase average
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

        # Individual points
        ax.scatter(phase, self.y, c="k", s=3, alpha=0.3, zorder=1,
                   label="Data")

        # Transit points
        if show_transit and self.x_transit is not None and self.y_transit is not None:
            phase_transit = (self.x_transit % period) / period
            ax.scatter(phase_transit, self.y_transit, c="C1", s=3,
                       alpha=0.4, zorder=1, label="Transit points")

        # Binned average
        valid = np.isfinite(bin_means)
        ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_errs[valid],
                    fmt="o", color="C3", ms=5, capsize=2, zorder=3,
                    label=f"Binned mean ({n_bins} bins)")

        # GP prediction phase-folded
        if show_gp and soln is not None:
            from celerite2 import GaussianProcess as celeriteGP
            from celerite2 import terms as celerite_terms

            s = soln

            def resolve(canonical):
                log_name = f"log_{canonical}"
                if log_name in s:
                    return float(np.exp(s[log_name]))
                if canonical in s:
                    return float(s[canonical])
                p = self._prior_params or {}
                if log_name in p and p[log_name]["type"] == "fixed":
                    return float(np.exp(p[log_name]["value"]))
                if canonical in p and p[canonical]["type"] == "fixed":
                    return float(p[canonical]["value"])
                raise KeyError(canonical)

            try:
                mean_val = resolve("mean")
                jitter = resolve("jitter")
                diag = self.yerr ** 2 + jitter ** 2
                kernel = self._build_kernel(resolve, celerite_terms, which="full")
                gp = celeriteGP(kernel, mean=mean_val)
                gp.compute(self.x, diag=diag)

                xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
                mu_gp, _ = gp.predict(self.y, t=xgrid, return_var=True)
                phase_gp = (xgrid % period) / period
                gp_sort = np.argsort(phase_gp)
                ax.plot(phase_gp[gp_sort], mu_gp[gp_sort], color="C0",
                        lw=1.5, alpha=0.7, zorder=2, label="GP prediction")
            except Exception:
                pass  # skip GP overlay if parameters are incomplete

        ax.set_xlim(0, 1)
        ax.set_xlabel("Phase", fontsize=16)
        ax.set_ylabel("Relative Flux", fontsize=16)
        ax.set_title(f"TIC {self.tic_id} — Sector {self.sector_label}  |  "f"Phase-folded (P = {period:.4f} d)", fontsize=18)
        ax.legend(fontsize=14)
        ax.minorticks_on()
        fig.tight_layout()

        fname = os.path.join(output_dir,f"TIC{self.tic_id}_sector{self.sector_label}_phase_fold.png",)
        fig.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"Saved phase-fold plot → {fname}\n")
        if return_fig:
            return fig, ax
        plt.close(fig)

    # ------------------------------------------------------------------
    # Reload saved results
    # ------------------------------------------------------------------

    def reload(self, sector, results_dir=None):
        """Reload saved GP fit results for a given sector.

        Parameters
        ----------
        sector : int
            Sector number.
        results_dir : str, optional
            Base results directory. Defaults to self.output_dir.

        Returns
        -------
        self
        """
        import pandas as pd

        if results_dir is None:
            results_dir = self.output_dir

        self.sector = sector
        sector_label = self.sector_label
        sector_tag = f"TIC{self.tic_id}_sector{sector_label}"

        # Load trace and summary
        trace_path = os.path.join(results_dir, "posterior_results", f"{sector_tag}.nc")
        self.trace = az.from_netcdf(trace_path)

        summary_path = os.path.join(results_dir, "summary_stats", f"{sector_tag}.csv")
        self.summary = pd.read_csv(summary_path, index_col=0)

        # Load data (same processing as fit used)
        self.load_data(sector)

        return self

    def compute_psd(self, freq_min=None, freq_max=None, n_freq=5000, log_spacing=True):
        """Compute the Lomb-Scargle power spectral density of the lightcurve.

        Parameters
        ----------
        freq_min : float, optional
            Minimum frequency in 1/days. Defaults to 1/(time span).
        freq_max : float, optional
            Maximum frequency in 1/days. Defaults to the Nyquist frequency.
        n_freq : int, optional
            Number of frequency points. Default is 5000.
        log_spacing : bool, optional
            If True, sample frequencies evenly in log space. Default is False.

        Returns
        -------
        freq : np.ndarray
            Frequency array in 1/days.
        period : np.ndarray
            Period array in days (sorted ascending).
        power : np.ndarray
            PSD at each frequency (same order as ``freq``).
        power_vs_period : np.ndarray
            PSD sorted by period (same order as ``period``).
        """
        from astropy.timeseries import LombScargle

        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        dt = np.median(np.diff(self.x))
        T = self.x.max() - self.x.min()
        if freq_min is None:
            freq_min = 1.0 / T
        if freq_max is None:
            freq_max = 0.5 / dt  # Nyquist

        if log_spacing:
            freq = 10**np.linspace(np.log10(freq_min), np.log10(freq_max), n_freq)
        else:
            freq = np.linspace(freq_min, freq_max, n_freq)
        period = 1.0 / freq[::-1]

        # Omit yerr so the PSD normalization matches the kernel PSD scaling
        ls = LombScargle(self.x, self.y)
        power = ls.power(freq, normalization="psd")
        power_vs_period = power[::-1]

        return freq, period, power, power_vs_period

    def plot_psd(self, output_dir=None, return_fig=False, show_kernel=True,
                 freq_min=None, freq_max=None, n_freq=5000, soln=None,
                 log_spacing=False):
        """Plot the power spectral density of the lightcurve.

        Uses ``compute_psd`` for the Lomb-Scargle periodogram. If a solution
        is available and ``show_kernel`` is True, the analytical GP kernel PSD
        is overlaid (SHOTerm and RotationTerm components shown individually).

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the plot. Defaults to
            ``self.output_dir/psd_plots/TIC{tic_id}``.
        return_fig : bool, optional
            If True, return ``(fig, ax)`` instead of closing the figure.
        show_kernel : bool, optional
            If True and a solution exists, overlay the analytical kernel
            PSD on the periodogram. Default is True.
        freq_min : float, optional
            Minimum frequency in 1/days. Defaults to 1/(time span).
        freq_max : float, optional
            Maximum frequency in 1/days. Defaults to the Nyquist frequency.
        n_freq : int, optional
            Number of frequency points. Default is 5000.
        soln : dict, optional
            Parameter solution dict to use. If None, uses ``self.map_soln``.
        log_spacing : bool, optional
            If True, sample frequencies evenly in log space. Default is False.

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes or None
            Returned only when ``return_fig`` is True.
        """
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

    def _draw_psd(self, ax, soln=None, show_kernel=True,
                  freq_min=None, freq_max=None, n_freq=5000, log_spacing=False):
        """Draw the PSD onto an existing axes."""
        freq, period, power, power_vs_period = self.compute_psd(
            freq_min=freq_min, freq_max=freq_max, n_freq=n_freq,
            log_spacing=log_spacing,
        )
        dt = np.median(np.diff(self.x))

        ax.loglog(freq, power, color="0.4", lw=0.8, alpha=0.7, label="Lomb-Scargle PSD")

        # --- Analytical kernel PSD from solution ---
        if soln is None:
            soln = self.map_soln
        if show_kernel and soln is not None:
            from celerite2 import terms as celerite_terms

            s = soln

            def resolve(canonical):
                log_name = f"log_{canonical}"
                if log_name in s:
                    return float(np.exp(s[log_name]))
                if canonical in s:
                    return float(s[canonical])
                p = self._prior_params or {}
                if log_name in p and p[log_name]["type"] == "fixed":
                    return float(np.exp(p[log_name]["value"]))
                if canonical in p and p[canonical]["type"] == "fixed":
                    return float(p[canonical]["value"])
                raise KeyError(
                    f"Parameter {canonical!r} not found in map_soln or _prior_params"
                )

            omega = 2 * np.pi * freq
            psd_scale = 4 * np.pi

            if getattr(self, "fit_shoterm", False):
                sho_kernel = self._build_kernel(resolve, celerite_terms, which="sho")
                sho_psd = psd_scale * sho_kernel.get_psd(omega)
                ax.loglog(freq, sho_psd, color="C2", lw=2, ls="--", label="SHOTerm (MAP)")

            if getattr(self, "fit_rotationterm", False):
                rot_kernel = self._build_kernel(resolve, celerite_terms, which="rot")
                rot_psd = psd_scale * rot_kernel.get_psd(omega)
                ax.loglog(freq, rot_psd, color="C0", lw=2, ls="--", label="RotationTerm (MAP)")

            if getattr(self, "fit_shoterm", False) and getattr(self, "fit_rotationterm", False):
                full_kernel = self._build_kernel(resolve, celerite_terms, which="full")
                full_psd = psd_scale * full_kernel.get_psd(omega)
                ax.loglog(freq, full_psd, color="C3", lw=2, label="Full kernel (MAP)")

            # Mark rotation period
            try:
                prot = resolve("period")
                ax.axvline(1.0 / prot, color="purple", ls=":", lw=2, alpha=0.8,
                           label=f"Prot = {prot:.2f} d")
                ax.axvline(2.0 / prot, color="pink", ls=":", lw=2, alpha=0.8,
                           label=f"2/Prot (1st harmonic)")
            except KeyError:
                pass

        # --- Mark Nyquist frequency ---
        nyquist_freq = 1.0 / (2.0 * dt)
        ax.axvline(nyquist_freq, color="C4", ls="-", lw=2, alpha=0.7)
        ax.text(
            nyquist_freq * 0.85, 0.95, f"Nyquist = {nyquist_freq:.1f} 1/d",
            transform=ax.get_xaxis_transform(),
            fontsize=18, color="C4", ha="right", va="top", rotation=90,
        )

        # --- Mark data cadence ---
        cadence_freq = 1.0 / dt
        ax.axvline(cadence_freq, color="C1", ls="--", lw=2, alpha=0.7)
        ax.text(
            cadence_freq * 0.85, 0.95, f"cadence = {dt*24*60:.1f} min",
            transform=ax.get_xaxis_transform(),
            fontsize=18, color="C1", ha="right", va="top", rotation=90,
        )

        ax.set_xlabel("Frequency [1/day]", fontsize=18)
        ax.set_ylabel("Power Spectral Density", fontsize=18)
        ax.set_title(f"TIC {self.tic_id} — Sector {self.sector_label} — PSD", fontsize=20)
        ax.legend(fontsize=16)
        ax.minorticks_on()

    def plot_acf(self, output_dir=None, return_fig=False, max_lag=None,
                 n_lags=500, show_kernel=True, soln=None):
        """Plot the autocorrelation function of the data and the GP kernel.

        The data ACF is estimated by interpolating onto a uniform grid and
        computing the normalized autocorrelation. If a solution exists and
        ``show_kernel`` is True, the analytical kernel autocovariance is
        overlaid (SHOTerm, RotationTerm, and full kernel shown individually,
        each normalized to unit peak).

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the plot. Defaults to
            ``self.output_dir/acf_plots/TIC{tic_id}``.
        return_fig : bool, optional
            If True, return ``(fig, ax)`` instead of closing the figure.
        max_lag : float, optional
            Maximum lag in days. Defaults to half the time span.
        n_lags : int, optional
            Number of lag points. Default is 500.
        show_kernel : bool, optional
            If True and a solution exists, overlay the kernel
            autocovariance function. Default is True.
        soln : dict, optional
            Parameter solution dict to use. If None, uses ``self.map_soln``.

        Returns
        -------
        matplotlib.figure.Figure, matplotlib.axes.Axes or None
            Returned only when ``return_fig`` is True.
        """
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

    def _draw_acf(self, ax, soln=None, show_kernel=True, max_lag=None,
                  n_lags=500):
        """Draw the ACF onto an existing axes."""
        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        T = self.x.max() - self.x.min()
        dt = np.median(np.diff(self.x))
        if max_lag is None:
            max_lag = T / 2.0

        # --- Data ACF via interpolation onto uniform grid ---
        t_uniform = np.arange(self.x.min(), self.x.max(), dt)
        y_uniform = np.interp(t_uniform, self.x, self.y)
        y_uniform -= np.mean(y_uniform)

        acf_full = np.correlate(y_uniform, y_uniform, mode="full")
        acf_full = acf_full[len(y_uniform) - 1:]
        acf_full /= acf_full[0]

        lag_times = np.arange(len(acf_full)) * dt
        mask = lag_times <= max_lag
        lag_times = lag_times[mask]
        acf_data = acf_full[mask]

        ax.plot(lag_times, acf_data, color="0.4", lw=2, alpha=0.7, label="Data ACF")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

        # --- Kernel autocovariance from solution ---
        if soln is None:
            soln = self.map_soln
        if show_kernel and soln is not None:
            from celerite2 import terms as celerite_terms

            s = soln

            def resolve(canonical):
                log_name = f"log_{canonical}"
                if log_name in s:
                    return float(np.exp(s[log_name]))
                if canonical in s:
                    return float(s[canonical])
                p = self._prior_params or {}
                if log_name in p and p[log_name]["type"] == "fixed":
                    return float(np.exp(p[log_name]["value"]))
                if canonical in p and p[canonical]["type"] == "fixed":
                    return float(p[canonical]["value"])
                raise KeyError(
                    f"Parameter {canonical!r} not found in map_soln or _prior_params"
                )

            tau = np.linspace(0, max_lag, n_lags)

            if getattr(self, "fit_shoterm", False):
                sho_kernel = self._build_kernel(resolve, celerite_terms, which="sho")
                sho_acf = sho_kernel.get_value(tau)
                sho_acf /= sho_acf[0]
                ax.plot(tau, sho_acf, color="C2", lw=2, ls="--", label="SHOTerm (MAP)")

            if getattr(self, "fit_rotationterm", False):
                rot_kernel = self._build_kernel(resolve, celerite_terms, which="rot")
                rot_acf = rot_kernel.get_value(tau)
                rot_acf /= rot_acf[0]
                ax.plot(tau, rot_acf, color="C0", lw=2, ls="--", label="RotationTerm (MAP)")

            if getattr(self, "fit_shoterm", False) and getattr(self, "fit_rotationterm", False):
                full_kernel = self._build_kernel(resolve, celerite_terms, which="full")
                full_acf = full_kernel.get_value(tau)
                full_acf /= full_acf[0]
                ax.plot(tau, full_acf, color="C3", lw=2, label="Full kernel (MAP)")

            # Mark rotation period
            try:
                prot = resolve("period")
                ax.axvline(prot, color="purple", ls=":", lw=2, alpha=0.8,
                           label=f"Prot = {prot:.2f} d")
                ax.axvline(2 * prot, color="pink", ls=":", lw=2, alpha=0.8,
                           label=f"2 Prot")
            except KeyError:
                pass

        ax.set_xlabel("Lag [days]", fontsize=16)
        ax.set_ylabel("Autocorrelation", fontsize=16)
        ax.set_title(f"TIC {self.tic_id} — Sector {self.sector_label} — ACF", fontsize=20)
        ax.set_xlim(0, max_lag)
        ax.legend(fontsize=16)
        ax.minorticks_on()

    def plot_psd_acf(self, output_dir=None, return_fig=False, soln=None, 
                     show_kernel=True, freq_min=None, freq_max=None,
                     n_freq=5000, log_spacing=False, max_lag=None, n_lags=500,
                     plot_suffix=""):
        """Two-panel plot combining PSD (top) and ACF (bottom).

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the plot. Defaults to
            ``self.output_dir/psd_acf_plots/TIC{tic_id}``.
        return_fig : bool, optional
            If True, return ``(fig, (ax_psd, ax_acf))`` instead of closing.
        soln : dict, optional
            Parameter solution dict. If None, uses ``self.map_soln``.
        show_kernel : bool, optional
            Overlay analytical kernel PSD/ACF. Default is True.
        freq_min, freq_max : float, optional
            Frequency bounds for the PSD panel.
        n_freq : int, optional
            Number of frequency points. Default is 5000.
        log_spacing : bool, optional
            Log-spaced frequencies for the PSD. Default is False.
        max_lag : float, optional
            Maximum lag in days for the ACF panel.
        n_lags : int, optional
            Number of lag points for the ACF. Default is 500.

        Returns
        -------
        matplotlib.figure.Figure, tuple of Axes or None
        """
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

    def fit_kernel_initial(self, max_lag=None, n_freq=2000, log_spacing=True,
                           acf_weight=1.0, psd_weight=1.0, method="L-BFGS-B",
                           verbose=True):
        """Fit kernel parameters by matching the data ACF and PSD.

        Uses ``scipy.optimize.minimize`` to find kernel parameters that
        best reproduce the empirical autocorrelation function and power
        spectral density.  The result is stored in ``self.map_soln`` so it
        can seed a subsequent ``find_map`` or MCMC run.

        Parameters
        ----------
        max_lag : float, optional
            Maximum lag in days for the ACF comparison.  Defaults to half
            the time span.
        n_freq : int, optional
            Number of frequency points for the PSD comparison.  Default 2000.
        log_spacing : bool, optional
            If True (default), sample frequencies evenly in log space.
        acf_weight : float, optional
            Relative weight for the ACF residuals in the cost function.
        psd_weight : float, optional
            Relative weight for the PSD residuals (in log space).
        method : str, optional
            Optimisation method passed to ``scipy.optimize.minimize``.
            Default is ``"L-BFGS-B"``.
        verbose : bool, optional
            If True, print the fitted parameter values.

        Returns
        -------
        dict
            Fitted parameter values (same format as ``map_soln``).
        """
        from scipy.optimize import minimize
        from celerite2 import terms as celerite_terms

        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        if self._prior_params is None:
            raise RuntimeError("Prior parameters must be set before fitting.")

        x, y, yerr = self.x, self.y, self.yerr
        dt = np.median(np.diff(x))
        T = x.max() - x.min()

        # --- Data ACF ---
        if max_lag is None:
            max_lag = T / 2.0
        t_uniform = np.arange(x.min(), x.max(), dt)
        y_uniform = np.interp(t_uniform, x, y)
        y_uniform -= np.mean(y_uniform)
        acf_full = np.correlate(y_uniform, y_uniform, mode="full")
        acf_full = acf_full[len(y_uniform) - 1:]
        acf_full /= acf_full[0]
        lag_times = np.arange(len(acf_full)) * dt
        lag_mask = lag_times <= max_lag
        tau_data = lag_times[lag_mask]
        acf_data = acf_full[lag_mask]

        # --- Data PSD ---
        freq, _, psd_data, _ = self.compute_psd(
            n_freq=n_freq, log_spacing=log_spacing,
        )
        log_psd_data = np.log10(np.maximum(psd_data, 1e-30))
        omega = 2 * np.pi * freq
        psd_scale = 4 * np.pi

        # --- Identify free kernel parameters and build mapping ---
        p = self._prior_params
        # Kernel-relevant parameters (exclude mean)
        kernel_params = [
            "log_jitter", "log_sigma", "log_rho", "Q", "log_Q",
            "period", "log_period", "log_sigma_rot", "log_Q0", "log_dQ", "f",
        ]
        free_names = []
        x0 = []
        bounds = []
        fixed_vals = {}

        for name in kernel_params:
            if name not in p:
                continue
            spec = p[name]
            if spec["type"] == "fixed":
                fixed_vals[name] = spec["value"]
            else:
                free_names.append(name)
                # Initial guess from prior
                if spec["type"] == "normal":
                    x0.append(spec["mu"])
                elif spec["type"] == "half_normal":
                    x0.append(spec["sigma"])
                elif spec["type"] == "uniform":
                    x0.append(0.5 * (spec["lower"] + spec["upper"]))
                elif spec["type"] == "truncated_normal":
                    x0.append(spec["mu"])
                elif spec["type"] == "log_normal":
                    x0.append(spec["mu"])
                else:
                    x0.append(0.0)

                # Bounds
                if spec["type"] == "uniform":
                    bounds.append((spec["lower"], spec["upper"]))
                elif spec["type"] == "truncated_normal":
                    bounds.append((spec["lower"], spec["upper"]))
                elif spec["type"] == "half_normal":
                    bounds.append((0.0, None))
                else:
                    bounds.append((None, None))

        # Also handle mean as fixed for kernel evaluation
        if "mean" in p:
            if p["mean"]["type"] == "fixed":
                fixed_vals["mean"] = p["mean"]["value"]
            else:
                fixed_vals["mean"] = np.mean(y)

        x0 = np.array(x0, dtype=np.float64)

        def params_to_dict(theta):
            """Map optimiser vector back to a parameter dict."""
            d = dict(fixed_vals)
            for i, name in enumerate(free_names):
                d[name] = theta[i]
            return d

        def resolve_from_dict(d):
            """Return a resolve callable for _build_kernel."""
            def resolve(canonical):
                log_name = f"log_{canonical}"
                if log_name in d:
                    return float(np.exp(d[log_name]))
                if canonical in d:
                    return float(d[canonical])
                raise KeyError(canonical)
            return resolve

        def cost(theta):
            d = params_to_dict(theta)
            try:
                resolve = resolve_from_dict(d)
                kernel = self._build_kernel(resolve, celerite_terms, which="full")

                # ACF cost: sum of squared differences
                model_acf = kernel.get_value(tau_data)
                model_acf_norm = model_acf / model_acf[0] if model_acf[0] != 0 else model_acf
                acf_cost = np.sum((model_acf_norm - acf_data) ** 2)

                # PSD cost: sum of squared log-differences
                model_psd = psd_scale * kernel.get_psd(omega)
                log_model_psd = np.log10(np.maximum(model_psd, 1e-30))
                psd_cost = np.sum((log_model_psd - log_psd_data) ** 2)

                return acf_weight * acf_cost + psd_weight * psd_cost
            except Exception:
                return 1e20

        result = minimize(cost, x0, method=method, bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12})

        # Build final parameter dict
        soln = params_to_dict(result.x)
        self.map_soln = soln

        # Set fit_shoterm / fit_rotationterm flags
        self.fit_shoterm = any(
            k in soln for k in ("log_sigma", "sigma", "log_rho", "rho")
        )
        self.fit_rotationterm = any(
            k in soln for k in ("period", "log_period")
        )

        if verbose:
            print(f"Initial kernel fit (success={result.success}, "
                  f"cost={result.fun:.4f}):")
            for name in free_names:
                print(f"  {name:20s} = {soln[name]:.6f}")
            for name, val in fixed_vals.items():
                if name != "mean":
                    print(f"  {name:20s} = {val:.6f} (fixed)")

        return soln
