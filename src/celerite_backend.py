import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import estimate_inverse_gamma_parameters

from .gp_fit_base import GPFit

__all__ = ["CeleriteGPFit"]

# ==================================================================
# CeleriteGPFit — celerite2 backend
# ==================================================================

class CeleriteGPFit(GPFit):
    """GP fitter using the celerite2 backend (PyMC + NumPyro samplers)."""

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
                log_name = f"log_{canonical}"
                if log_name in rv:
                    return pt.exp(rv[log_name])
                elif canonical in rv:
                    return rv[canonical]
                raise KeyError(f"Neither {log_name!r} nor {canonical!r} found in prior_params")

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
                raise ValueError("No kernel terms enabled.")

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

    def build_numpyro_model(self, prior_params=None):
        """Build a NumPyro GP model using celerite2.jax."""
        import jax
        jax.config.update("jax_platforms", "cpu")

        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp
        from celerite2.jax import GaussianProcess as JaxGP
        from celerite2.jax import terms as jax_terms

        x, y, yerr = self.x, self.y, self.yerr

        if prior_params is not None:
            self._prior_params = prior_params
        p = self._prior_params

        def model_fn():
            rv = {}
            for name, spec in p.items():
                if spec["type"] == "normal":
                    rv[name] = numpyro.sample(name, dist.Normal(spec["mu"], spec["sigma"]))
                elif spec["type"] == "half_normal":
                    rv[name] = numpyro.sample(name, dist.HalfNormal(spec["sigma"]))
                elif spec["type"] == "log_normal":
                    rv[name] = numpyro.sample(name, dist.LogNormal(spec["mu"], spec["sigma"]))
                elif spec["type"] == "truncated_normal":
                    rv[name] = numpyro.sample(
                        name,
                        dist.TruncatedNormal(
                            loc=spec["mu"], scale=spec["sigma"],
                            low=spec["lower"], high=spec["upper"],
                        ),
                    )
                elif spec["type"] == "uniform":
                    rv[name] = numpyro.sample(name, dist.Uniform(spec["lower"], spec["upper"]))
                elif spec["type"] == "inverse_gamma":
                    ig = estimate_inverse_gamma_parameters(spec["lower"], spec["upper"])
                    rv[name] = numpyro.sample(
                        name, dist.InverseGamma(ig["alpha"], ig["beta"])
                    )
                elif spec["type"] == "fixed":
                    rv[name] = spec["value"]
                else:
                    raise ValueError(f"Unknown prior type {spec['type']!r} for parameter {name!r}")

            def resolve(canonical):
                log_name = f"log_{canonical}"
                if log_name in rv:
                    return jnp.exp(rv[log_name])
                elif canonical in rv:
                    return rv[canonical]
                raise KeyError(f"Neither {log_name!r} nor {canonical!r} found in prior_params")

            kernel = self._build_kernel(resolve, jax_terms)
            jitter = resolve("jitter")
            gp = JaxGP(kernel, mean=resolve("mean"))
            gp.compute(x, diag=yerr**2 + jitter**2)
            numpyro.factor("gp", gp.log_likelihood(y))

        self._numpyro_model = model_fn
        return model_fn

    # ------------------------------------------------------------------
    # Kernel building
    # ------------------------------------------------------------------

    def _build_kernel(self, resolve, terms_module=None, which="full"):
        """Build a celerite2 kernel using a ``resolve(canonical)`` callable."""
        if terms_module is None:
            from celerite2 import terms as terms_module

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

    # ------------------------------------------------------------------
    # MAP optimization
    # ------------------------------------------------------------------

    def find_map(self, sector, sampler="pymc", start=None):
        """Find the MAP solution for a sector."""
        if self.sector != sector or self.x is None:
            self.load_data(sector)

        if not hasattr(self, "_prior_params") or self._prior_params is None:
            raise ValueError("Prior parameters must be set before finding MAP.")

        if sampler == "pymc":
            model = self.build_pymc_model()
            self._model = model
            self.map_soln = self._find_map_pymc(model, start=start)
        elif sampler == "numpyro":
            model = self.build_pymc_model()
            self._model = model
            self.map_soln = self._find_map_pymc(model, start=start)
            self.build_numpyro_model()
        else:
            raise NotImplementedError("MAP optimization with pymc3 is not yet implemented.")

        return self.map_soln

    def _find_map_pymc(self, model, start=None):
        """Find MAP using pm.find_MAP (three-stage)."""
        import pymc as pm

        free = set(self.free_var_names)
        use_log_period = "log_period" in free

        def get_vars(names):
            return [model.named_vars[n] for n in names if n in free and n in model.named_vars]

        def filter_point(point):
            return {k: v for k, v in point.items() if k in free}

        def clamp_bounded(point):
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
            start = {k: v for k, v in start.items() if k in free}

        stage2 = [n for n in self.free_var_names if n not in {"mean", period_key}]

        with model:
            if stage1:
                map_soln = clamp_bounded(filter_point(pm.find_MAP(start=start, vars=get_vars(stage1))))
            else:
                map_soln = {}

            if stage2:
                map_soln = clamp_bounded(filter_point(pm.find_MAP(start=map_soln, vars=get_vars(stage2))))

            map_soln = clamp_bounded(filter_point(pm.find_MAP(start=map_soln, vars=get_vars(self.free_var_names))))

        return map_soln

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_pymc(self, model, tune, draws, chains, cores, target_accept,
                     map_soln=None, callback=None, step_scale=None, init_jitter=0.1):
        """Run MCMC sampling with the PyMC NUTS sampler."""
        import pymc as pm

        if map_soln is None:
            map_soln = self._find_map_pymc(model)

        initvals = self._prepare_init_points(map_soln, chains, init_jitter)
        if initvals is None:
            initvals = map_soln

        with model:
            sample_kwargs = dict(
                tune=tune,
                draws=draws,
                chains=chains,
                cores=cores,
                init="adapt_full",
                initvals=initvals,
                target_accept=target_accept,
                return_inferencedata=True,
                callback=callback,
            )
            if step_scale is not None:
                sample_kwargs["step_scale"] = step_scale

            trace = pm.sample(**sample_kwargs)

        return trace

    def fit(self, x=None, y=None, yerr=None, sampler="pymc", tune=2000, draws=1000, chains=4,
            target_accept=0.9, cores=1, chain_method="parallel", map_soln=None,
            callback=None, step_scale=None, init_jitter=0.1):
        """Fit a GP model using celerite2 backend."""
        if (x is None) or (y is None) or (yerr is None):
            x, y, yerr = self.load_data(self.sector)

        if map_soln is None:
            map_soln = {}

        if sampler == "pymc":
            model = getattr(self, "_model", None) or self.build_pymc_model()
            self.trace = self._sample_pymc(
                model, tune=tune, draws=draws, chains=chains,
                cores=cores, target_accept=target_accept,
                map_soln=map_soln, callback=callback,
                step_scale=step_scale, init_jitter=init_jitter
            )
        elif sampler == "numpyro":
            model_fn = getattr(self, "_numpyro_model", None) or self.build_numpyro_model()
            self.trace = self._sample_numpyro(
                model_fn, tune=tune, draws=draws, chains=chains,
                target_accept=target_accept, chain_method=chain_method,
                map_soln=map_soln, step_scale=step_scale,
                init_jitter=init_jitter,
            )
        else:
            raise NotImplementedError(f"Sampler {sampler!r} is not supported. Use 'pymc' or 'numpyro'.")

        self._save_results()
        self._plot_results()
        return self.trace

    # ------------------------------------------------------------------
    # GP prediction
    # ------------------------------------------------------------------

    def predict(self, xgrid=None):
        """Compute GP mean prediction from posterior samples."""
        from celerite2 import GaussianProcess as celeriteGP
        from celerite2 import terms as celerite_terms

        if xgrid is None:
            xgrid = self.x

        posterior = self.trace.posterior
        flat = posterior.stack(sample=("chain", "draw"))

        n_samples = flat.sizes["sample"]
        preds = np.empty((n_samples, len(xgrid)))

        def resolve(canonical, idx):
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
        """Compute GP prediction using the MAP solution."""
        from celerite2 import GaussianProcess as celeriteGP
        from celerite2 import terms as celerite_terms

        if self.map_soln is None:
            return None

        if xgrid is None:
            xgrid = self.x

        resolve = self._resolve_from_soln(self.map_soln)
        kernel = self._build_kernel(resolve, celerite_terms)
        jitter = resolve("jitter")
        gp = celeriteGP(kernel, mean=resolve("mean"))
        gp.compute(self.x, diag=self.yerr**2 + jitter**2)

        return gp.predict(self.y, t=xgrid, return_var=True)

    # ------------------------------------------------------------------
    # Kernel component plots
    # ------------------------------------------------------------------

    def plot_kernel_components(self, output_dir=None, soln=None):
        """Plot lightcurve with individual GP kernel components."""
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

        resolve = self._resolve_from_soln(soln)
        mean_val = resolve("mean")
        jitter = resolve("jitter")
        diag = self.yerr**2 + jitter**2
        xgrid = np.linspace(self.x.min(), self.x.max(), 2000)

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

        self._draw_kernel_component_figure(predictions, colors, soln, xgrid, output_dir)

    # ------------------------------------------------------------------
    # Phase fold
    # ------------------------------------------------------------------

    def plot_phase_fold(self, period=None, n_bins=50, output_dir=None,
                        return_fig=False, soln=None, show_gp=False, show_transit=False):
        """Plot the phase-folded lightcurve at a given period."""
        fig, ax, soln, period, output_dir, return_fig = self._plot_phase_fold_common(
            period, n_bins, output_dir, return_fig, soln, show_gp, show_transit)

        if show_gp and soln is not None:
            from celerite2 import GaussianProcess as celeriteGP
            from celerite2 import terms as celerite_terms

            try:
                resolve = self._resolve_from_soln(soln)
                kernel = self._build_kernel(resolve, celerite_terms, which="full")
                jitter = resolve("jitter")
                gp = celeriteGP(kernel, mean=resolve("mean"))
                gp.compute(self.x, diag=self.yerr**2 + jitter**2)

                xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
                mu_gp, _ = gp.predict(self.y, t=xgrid, return_var=True)
                phase_gp = (xgrid % period) / period
                gp_sort = np.argsort(phase_gp)
                ax.plot(phase_gp[gp_sort], mu_gp[gp_sort], color="C0",
                        lw=1.5, alpha=0.7, zorder=2, label="GP prediction")
            except Exception:
                pass

        return self._finalize_phase_fold(fig, ax, period, output_dir, return_fig)

    # ------------------------------------------------------------------
    # PSD / ACF drawing (kernel overlay)
    # ------------------------------------------------------------------

    def _draw_psd(self, ax, soln=None, show_kernel=True,
                  freq_min=None, freq_max=None, n_freq=5000, log_spacing=False):
        """Draw the PSD onto an existing axes."""
        freq, period, power, power_vs_period = self.compute_psd(
            freq_min=freq_min, freq_max=freq_max, n_freq=n_freq,
            log_spacing=log_spacing,
        )
        dt = np.median(np.diff(self.x))

        ax.loglog(freq, power, color="0.4", lw=0.8, alpha=0.7, label="Lomb-Scargle PSD")

        if soln is None:
            soln = self.map_soln
        if show_kernel and soln is not None:
            from celerite2 import terms as celerite_terms

            resolve = self._resolve_from_soln(soln)
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

            try:
                prot = resolve("period")
                ax.axvline(1.0 / prot, color="purple", ls=":", lw=2, alpha=0.8,
                           label=f"Prot = {prot:.2f} d")
                ax.axvline(2.0 / prot, color="pink", ls=":", lw=2, alpha=0.8,
                           label=f"2/Prot (1st harmonic)")
            except KeyError:
                pass

        nyquist_freq = 1.0 / (2.0 * dt)
        ax.axvline(nyquist_freq, color="C4", ls="-", lw=2, alpha=0.7)
        ax.text(nyquist_freq * 0.85, 0.95, f"Nyquist = {nyquist_freq:.1f} 1/d",
                transform=ax.get_xaxis_transform(),
                fontsize=18, color="C4", ha="right", va="top", rotation=90)

        cadence_freq = 1.0 / dt
        ax.axvline(cadence_freq, color="C1", ls="--", lw=2, alpha=0.7)
        ax.text(cadence_freq * 0.85, 0.95, f"cadence = {dt*24*60:.1f} min",
                transform=ax.get_xaxis_transform(),
                fontsize=18, color="C1", ha="right", va="top", rotation=90)

        ax.set_xlabel("Frequency [1/day]", fontsize=18)
        ax.set_ylabel("Power Spectral Density", fontsize=18)
        ax.set_title(f"TIC {self.tic_id} — Sector {self.sector_label} — PSD", fontsize=20)
        ax.legend(fontsize=16)
        ax.minorticks_on()

    def _draw_acf(self, ax, soln=None, show_kernel=True, max_lag=None,
                  n_lags=500):
        """Draw the ACF onto an existing axes."""
        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        T = self.x.max() - self.x.min()
        dt = np.median(np.diff(self.x))
        if max_lag is None:
            max_lag = T / 2.0

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

        if soln is None:
            soln = self.map_soln
        if show_kernel and soln is not None:
            from celerite2 import terms as celerite_terms

            resolve = self._resolve_from_soln(soln)
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

    # ------------------------------------------------------------------
    # Initial kernel fit
    # ------------------------------------------------------------------

    def fit_kernel_initial(self, max_lag=None, n_freq=2000, log_spacing=True,
                           acf_weight=1.0, psd_weight=1.0, method="L-BFGS-B",
                           verbose=True):
        """Fit kernel parameters by matching the data ACF and PSD."""
        from scipy.optimize import minimize
        from celerite2 import terms as celerite_terms

        if self.x is None or self.y is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        if self._prior_params is None:
            raise RuntimeError("Prior parameters must be set before fitting.")

        x, y, yerr = self.x, self.y, self.yerr
        dt = np.median(np.diff(x))
        T = x.max() - x.min()

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

        freq, _, psd_data, _ = self.compute_psd(n_freq=n_freq, log_spacing=log_spacing)
        log_psd_data = np.log10(np.maximum(psd_data, 1e-30))
        omega = 2 * np.pi * freq
        psd_scale = 4 * np.pi

        p = self._prior_params
        kernel_params = [
            "log_jitter", "log_sigma", "log_rho", "Q", "log_Q",
            "period", "log_period", "log_sigma_rot", "log_Q0", "log_dQ", "f",
        ]
        free_names, x0, bounds, fixed_vals = [], [], [], {}

        for name in kernel_params:
            if name not in p:
                continue
            spec = p[name]
            if spec["type"] == "fixed":
                fixed_vals[name] = spec["value"]
            else:
                free_names.append(name)
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

                if spec["type"] == "uniform":
                    bounds.append((spec["lower"], spec["upper"]))
                elif spec["type"] == "truncated_normal":
                    bounds.append((spec["lower"], spec["upper"]))
                elif spec["type"] == "half_normal":
                    bounds.append((0.0, None))
                else:
                    bounds.append((None, None))

        if "mean" in p:
            if p["mean"]["type"] == "fixed":
                fixed_vals["mean"] = p["mean"]["value"]
            else:
                fixed_vals["mean"] = np.mean(y)

        x0 = np.array(x0, dtype=np.float64)

        def params_to_dict(theta):
            d = dict(fixed_vals)
            for i, name in enumerate(free_names):
                d[name] = theta[i]
            return d

        def cost(theta):
            d = params_to_dict(theta)
            try:
                resolve = self._resolve_from_soln(d)
                kernel = self._build_kernel(resolve, celerite_terms, which="full")

                model_acf = kernel.get_value(tau_data)
                model_acf_norm = model_acf / model_acf[0] if model_acf[0] != 0 else model_acf
                acf_cost = np.sum((model_acf_norm - acf_data) ** 2)

                model_psd = psd_scale * kernel.get_psd(omega)
                log_model_psd = np.log10(np.maximum(model_psd, 1e-30))
                psd_cost = np.sum((log_model_psd - log_psd_data) ** 2)

                return acf_weight * acf_cost + psd_weight * psd_cost
            except Exception:
                return 1e20

        result = minimize(cost, x0, method=method, bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12})

        soln = params_to_dict(result.x)
        self.map_soln = soln

        self.fit_shoterm = any(k in soln for k in ("log_sigma", "sigma", "log_rho", "rho"))
        self.fit_rotationterm = any(k in soln for k in ("period", "log_period"))

        if verbose:
            print(f"Initial kernel fit (success={result.success}, cost={result.fun:.4f}):")
            for name in free_names:
                print(f"  {name:20s} = {soln[name]:.6f}")
            for name, val in fixed_vals.items():
                if name != "mean":
                    print(f"  {name:20s} = {val:.6f} (fixed)")

        return soln

