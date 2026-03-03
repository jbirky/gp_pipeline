import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import estimate_inverse_gamma_parameters

from .gp_fit_base import GPFit

__all__ = ["TinygpGPFit"]

# ------------------------------------------------------------------
# tinygp helpers (used by TinygpGPFit)
# ------------------------------------------------------------------

def _rotation_term(sigma, period, Q0, dQ, f):
    """Build a RotationTerm-equivalent from two tinygp SHO kernels.

    Uses jax.numpy so that this works with both plain floats and JAX traced arrays.
    """
    import jax.numpy as jnp
    from tinygp.kernels.quasisep import SHO

    Q1 = 0.5 + Q0 + dQ
    Q2 = 0.5 + Q0
    w1 = 4 * jnp.pi * Q1 / (period * jnp.sqrt(4 * Q1**2 - 1))
    w2 = 8 * jnp.pi * Q2 / (period * jnp.sqrt(4 * Q2**2 - 1))
    sigma1 = sigma / jnp.sqrt(1 + f)
    sigma2 = sigma * jnp.sqrt(f / (1 + f))
    return SHO(omega=w1, quality=Q1, sigma=sigma1) + SHO(omega=w2, quality=Q2, sigma=sigma2)


def _sho_psd(sigma, omega0, quality, w):
    r"""Analytical PSD of a single SHO kernel.

    .. math::
        S(\omega) = \sqrt{\frac{2}{\pi}}\,
        \frac{\sigma^2\,\omega_0^4 / Q}
             {(\omega^2 - \omega_0^2)^2 + \omega_0^2\,\omega^2 / Q^2}
    """
    w = np.asarray(w, dtype=np.float64)
    S0 = sigma**2 / (omega0 * quality)
    return np.sqrt(2 / np.pi) * S0 * omega0**4 / (
        (w**2 - omega0**2) ** 2 + omega0**2 * w**2 / quality**2
    )


def _sho_acf(sigma, omega0, quality, tau):
    r"""Analytical autocovariance of a single SHO kernel."""
    tau = np.asarray(tau, dtype=np.float64)
    S0 = sigma**2 / (omega0 * quality)

    if quality > 0.5:
        eta = omega0 / (2 * quality)
        omega_d = omega0 * np.sqrt(1 - 1 / (4 * quality**2))
        return (
            sigma**2
            * np.exp(-eta * tau)
            * (np.cos(omega_d * tau) + eta / omega_d * np.sin(omega_d * tau))
        )
    elif quality == 0.5:
        eta = omega0
        return sigma**2 * np.exp(-eta * tau) * (1 + eta * tau)
    else:
        eta = omega0 / (2 * quality)
        delta = omega0 * np.sqrt(1 / (4 * quality**2) - 1)
        return (
            sigma**2
            * np.exp(-eta * tau)
            * (np.cosh(delta * tau) + eta / delta * np.sinh(delta * tau))
        )

# ==================================================================
# TinygpGPFit — tinygp (JAX) backend
# ==================================================================

class TinygpGPFit(GPFit):
    """GP fitter using the tinygp backend (pure JAX, GPU-compatible)."""

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def build_pymc_model(self, prior_params=None):
        """Parse prior parameters and detect kernel flags (no PyMC model built)."""
        if prior_params is not None:
            self._prior_params = prior_params
        self._detect_kernel_flags()

    def build_numpyro_model(self, prior_params=None):
        """Build a NumPyro GP model using tinygp (pure JAX, GPU-compatible)."""
        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp
        import tinygp

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

            kernel = self._build_kernel(resolve)
            jitter = resolve("jitter")
            mean_val = resolve("mean")
            gp = tinygp.GaussianProcess(
                kernel, X=x, diag=yerr**2 + jitter**2, mean=mean_val
            )
            numpyro.factor("gp", gp.log_probability(y))

        self._numpyro_model = model_fn
        return model_fn

    # ------------------------------------------------------------------
    # Kernel building
    # ------------------------------------------------------------------

    def _build_kernel(self, resolve, which="full"):
        """Build a tinygp kernel using a ``resolve(canonical)`` callable."""
        from tinygp.kernels.quasisep import SHO

        kernel = None

        if which in ("full", "sho") and getattr(self, "fit_shoterm", True):
            rho = resolve("rho")
            omega = 2 * np.pi / rho
            kernel = SHO(
                omega=omega,
                quality=resolve("Q"),
                sigma=resolve("sigma"),
            )

        if which in ("full", "rot") and getattr(self, "fit_rotationterm", True):
            rot = _rotation_term(
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

    def _build_kernel_jax(self, resolve, which="full"):
        """Build a tinygp kernel using JAX arrays (for JIT-compiled MAP)."""
        import jax.numpy as jnp
        from tinygp.kernels.quasisep import SHO

        kernel = None

        if which in ("full", "sho") and getattr(self, "fit_shoterm", True):
            rho = resolve("rho")
            omega = 2 * jnp.pi / rho
            kernel = SHO(
                omega=omega,
                quality=resolve("Q"),
                sigma=resolve("sigma"),
            )

        if which in ("full", "rot") and getattr(self, "fit_rotationterm", True):
            sigma_rot = resolve("sigma_rot")
            period = resolve("period")
            Q0 = resolve("Q0")
            dQ = resolve("dQ")
            f = resolve("f")

            Q1 = 0.5 + Q0 + dQ
            Q2 = 0.5 + Q0
            w1 = 4 * jnp.pi * Q1 / (period * jnp.sqrt(4 * Q1**2 - 1))
            w2 = 8 * jnp.pi * Q2 / (period * jnp.sqrt(4 * Q2**2 - 1))
            sigma1 = sigma_rot / jnp.sqrt(1 + f)
            sigma2 = sigma_rot * jnp.sqrt(f / (1 + f))
            rot = SHO(omega=w1, quality=Q1, sigma=sigma1) + SHO(omega=w2, quality=Q2, sigma=sigma2)
            kernel = kernel + rot if kernel is not None else rot

        if kernel is None:
            raise RuntimeError("No kernel components are enabled.")
        return kernel

    # ------------------------------------------------------------------
    # MAP optimization
    # ------------------------------------------------------------------

    def find_map(self, sector, sampler="numpyro", start=None):
        """Find the MAP solution for a sector using JAX optimization."""
        if self.sector != sector or self.x is None:
            self.load_data(sector)

        if not hasattr(self, "_prior_params") or self._prior_params is None:
            raise ValueError("Prior parameters must be set before finding MAP.")

        if not hasattr(self, "fit_shoterm"):
            self.build_pymc_model()

        self.map_soln = self._find_map_jax(start=start)
        self.build_numpyro_model()
        return self.map_soln

    def _find_map_jax(self, start=None):
        """Find MAP using JAX + scipy L-BFGS-B (three-stage)."""
        import jax
        import jax.numpy as jnp
        from scipy.optimize import minimize
        import tinygp
        from tinygp.kernels.quasisep import SHO

        p = self._prior_params
        free = self.free_var_names
        x_data = jnp.array(self.x)
        y_data = jnp.array(self.y)
        yerr_data = jnp.array(self.yerr)

        use_log_period = "log_period" in free

        param_bounds = {}
        for name in free:
            spec = p[name]
            if spec["type"] == "uniform":
                eps = 1e-6 * (spec["upper"] - spec["lower"])
                param_bounds[name] = (spec["lower"] + eps, spec["upper"] - eps)
            elif spec["type"] == "truncated_normal":
                eps = 1e-6 * (spec["upper"] - spec["lower"])
                param_bounds[name] = (spec["lower"] + eps, spec["upper"] - eps)
            elif spec["type"] == "half_normal":
                param_bounds[name] = (1e-10, None)
            else:
                param_bounds[name] = (None, None)

        def _log_prior(name, val):
            spec = p[name]
            t = spec["type"]
            if t == "normal":
                return -0.5 * ((val - spec["mu"]) / spec["sigma"]) ** 2
            elif t == "truncated_normal":
                in_bounds = (val >= spec["lower"]) & (val <= spec["upper"])
                lp = -0.5 * ((val - spec["mu"]) / spec["sigma"]) ** 2
                return jnp.where(in_bounds, lp, -jnp.inf)
            elif t == "uniform":
                in_bounds = (val >= spec["lower"]) & (val <= spec["upper"])
                return jnp.where(in_bounds, 0.0, -jnp.inf)
            elif t == "half_normal":
                return jnp.where(val >= 0, -0.5 * (val / spec["sigma"]) ** 2, -jnp.inf)
            elif t == "log_normal":
                return jnp.where(
                    val > 0,
                    -0.5 * ((jnp.log(val) - spec["mu"]) / spec["sigma"]) ** 2 - jnp.log(val),
                    -jnp.inf,
                )
            elif t == "inverse_gamma":
                ig = estimate_inverse_gamma_parameters(spec["lower"], spec["upper"])
                alpha, beta = ig["alpha"], ig["beta"]
                return jnp.where(
                    val > 0,
                    alpha * jnp.log(beta) - (alpha + 1) * jnp.log(val) - beta / val,
                    -jnp.inf,
                )
            return 0.0

        def _resolve_from_array(theta, names):
            vals = {n: theta[i] for i, n in enumerate(names)}
            for n, spec in p.items():
                if spec["type"] == "fixed":
                    vals[n] = spec["value"]

            def resolve(canonical):
                log_name = f"log_{canonical}"
                if log_name in vals:
                    return jnp.exp(vals[log_name])
                elif canonical in vals:
                    return vals[canonical]
                raise KeyError(f"Neither {log_name!r} nor {canonical!r} found")
            return resolve, vals

        def neg_log_posterior(theta, opt_names):
            resolve, vals = _resolve_from_array(theta, opt_names)

            kernel = self._build_kernel_jax(resolve)
            jitter = resolve("jitter")
            mean_val = resolve("mean")
            gp = tinygp.GaussianProcess(
                kernel, X=x_data, diag=yerr_data**2 + jitter**2, mean=mean_val
            )
            ll = gp.log_probability(y_data)

            lp = 0.0
            for n in opt_names:
                lp = lp + _log_prior(n, vals[n])

            return -(ll + lp)

        def _optimize_stage(opt_names, current_soln):
            if not opt_names:
                return current_soln

            x0 = jnp.array([float(current_soln.get(n, 0.0)) for n in opt_names])
            bounds_list = [param_bounds.get(n, (None, None)) for n in opt_names]

            val_and_grad_fn = jax.jit(jax.value_and_grad(lambda theta: neg_log_posterior(theta, opt_names)))

            def scipy_objective(x_np):
                theta = jnp.array(x_np)
                val, grad = val_and_grad_fn(theta)
                v = float(val)
                g = np.array(grad, dtype=np.float64)
                if not np.isfinite(v):
                    v = 1e20
                    g = np.zeros_like(g)
                return v, g

            result = minimize(
                scipy_objective, np.array(x0), method="L-BFGS-B",
                jac=True, bounds=bounds_list,
                options={"maxiter": 2000, "ftol": 1e-12},
            )

            updated = dict(current_soln)
            for i, n in enumerate(opt_names):
                updated[n] = float(result.x[i])
            return updated

        # Build initial solution
        if start is None:
            start = {}
        soln = {}
        for name in free:
            if name in start:
                soln[name] = float(start[name])
            else:
                spec = p[name]
                if name == "mean":
                    soln[name] = float(np.mean(self.y))
                elif name == "log_period":
                    soln[name] = float(np.log(self.prot_init))
                elif name == "period":
                    soln[name] = float(self.prot_init)
                elif spec["type"] == "normal":
                    soln[name] = float(spec["mu"])
                elif spec["type"] == "truncated_normal":
                    soln[name] = float(spec["mu"])
                elif spec["type"] == "uniform":
                    soln[name] = float(0.5 * (spec["lower"] + spec["upper"]))
                elif spec["type"] == "half_normal":
                    soln[name] = float(spec["sigma"])
                elif spec["type"] == "log_normal":
                    soln[name] = float(spec["mu"])
                elif spec["type"] == "inverse_gamma":
                    soln[name] = float(0.5 * (spec["lower"] + spec["upper"]))
                else:
                    soln[name] = 0.0

        period_key = "log_period" if use_log_period else "period"
        stage1 = [n for n in ["mean", period_key] if n in free]
        soln = _optimize_stage(stage1, soln)

        stage2 = [n for n in free if n not in {"mean", period_key}]
        soln = _optimize_stage(stage2, soln)

        soln = _optimize_stage(free, soln)
        return soln

    # ------------------------------------------------------------------
    # Sampling / Fitting
    # ------------------------------------------------------------------

    def fit(self, x=None, y=None, yerr=None, sampler="numpyro", tune=2000, draws=1000, chains=4,
            target_accept=0.9, cores=1, chain_method="parallel", map_soln=None,
            callback=None, step_scale=None, init_jitter=0.1):
        """Fit a GP model using tinygp backend (NumPyro only)."""
        if (x is None) or (y is None) or (yerr is None):
            x, y, yerr = self.load_data(self.sector)

        if map_soln is None:
            map_soln = {}

        if sampler == "numpyro":
            model_fn = getattr(self, "_numpyro_model", None) or self.build_numpyro_model()
            self.trace = self._sample_numpyro(
                model_fn, tune=tune, draws=draws, chains=chains,
                target_accept=target_accept, chain_method=chain_method,
                map_soln=map_soln, step_scale=step_scale,
                init_jitter=init_jitter,
            )
        else:
            raise NotImplementedError(
                f"Sampler {sampler!r} is not supported in the tinygp backend. "
                f"Use sampler='numpyro'."
            )

        self._save_results()
        self._plot_results()
        return self.trace

    # ------------------------------------------------------------------
    # GP prediction
    # ------------------------------------------------------------------

    def predict(self, xgrid=None):
        """Compute GP mean prediction from posterior samples."""
        import tinygp

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
            kernel = self._build_kernel(lambda c: resolve(c, i))
            jitter = resolve("jitter", i)
            mean_val = resolve("mean", i)
            gp = tinygp.GaussianProcess(
                kernel, X=self.x, diag=self.yerr**2 + jitter**2, mean=mean_val
            )
            _, cond = gp.condition(self.y, X_test=xgrid)
            preds[i] = cond.loc

        return preds.mean(axis=0), preds.var(axis=0)

    def predict_map(self, xgrid=None):
        """Compute GP prediction using the MAP solution."""
        import tinygp

        if self.map_soln is None:
            return None

        if xgrid is None:
            xgrid = self.x

        resolve = self._resolve_from_soln(self.map_soln)
        kernel = self._build_kernel(resolve)
        jitter = resolve("jitter")
        mean_val = resolve("mean")
        gp = tinygp.GaussianProcess(
            kernel, X=self.x, diag=self.yerr**2 + jitter**2, mean=mean_val
        )
        _, cond = gp.condition(self.y, X_test=xgrid)
        return cond.loc, cond.variance

    # ------------------------------------------------------------------
    # Kernel component plots
    # ------------------------------------------------------------------

    def plot_kernel_components(self, output_dir=None, soln=None):
        """Plot lightcurve with individual GP kernel components."""
        import tinygp

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
            sho_kernel = self._build_kernel(resolve, which="sho")
            gp = tinygp.GaussianProcess(sho_kernel, X=self.x, diag=diag, mean=mean_val)
            _, cond = gp.condition(self.y, X_test=xgrid)
            predictions["SHOTerm"] = (np.asarray(cond.loc), np.sqrt(np.asarray(cond.variance)))
            colors["SHOTerm"] = "C2"

        if getattr(self, "fit_rotationterm", False):
            rot_kernel = self._build_kernel(resolve, which="rot")
            gp = tinygp.GaussianProcess(rot_kernel, X=self.x, diag=diag, mean=mean_val)
            _, cond = gp.condition(self.y, X_test=xgrid)
            predictions["RotationTerm"] = (np.asarray(cond.loc), np.sqrt(np.asarray(cond.variance)))
            colors["RotationTerm"] = "C0"

        if getattr(self, "fit_shoterm", False) and getattr(self, "fit_rotationterm", False):
            full_kernel = self._build_kernel(resolve, which="full")
            gp = tinygp.GaussianProcess(full_kernel, X=self.x, diag=diag, mean=mean_val)
            _, cond = gp.condition(self.y, X_test=xgrid)
            predictions["SHOTerm + RotationTerm"] = (np.asarray(cond.loc), np.sqrt(np.asarray(cond.variance)))
            colors["SHOTerm + RotationTerm"] = "C3"

        # Reuse the shared figure layout from GPFit base class
        self._draw_kernel_component_figure(predictions, colors, soln, xgrid, output_dir)

    # ------------------------------------------------------------------
    # Phase fold
    # ------------------------------------------------------------------

    def plot_phase_fold(self, period=None, n_bins=50, output_dir=None,
                        return_fig=False, soln=None, show_gp=False, show_transit=False):
        """Plot the phase-folded lightcurve at a given period."""
        import tinygp

        fig, ax, soln, period, output_dir, return_fig = self._plot_phase_fold_common(
            period, n_bins, output_dir, return_fig, soln, show_gp, show_transit)

        if show_gp and soln is not None:
            try:
                resolve = self._resolve_from_soln(soln)
                kernel = self._build_kernel(resolve, which="full")
                jitter = resolve("jitter")
                gp = tinygp.GaussianProcess(kernel, X=self.x, diag=self.yerr**2 + jitter**2, mean=resolve("mean"))

                xgrid = np.linspace(self.x.min(), self.x.max(), 2000)
                _, cond = gp.condition(self.y, X_test=xgrid)
                mu_gp = np.asarray(cond.loc)
                phase_gp = (xgrid % period) / period
                gp_sort = np.argsort(phase_gp)
                ax.plot(phase_gp[gp_sort], mu_gp[gp_sort], color="C0",
                        lw=1.5, alpha=0.7, zorder=2, label="GP prediction")
            except Exception:
                pass

        return self._finalize_phase_fold(fig, ax, period, output_dir, return_fig)

    # ------------------------------------------------------------------
    # SHO parameter extraction (for analytical PSD/ACF)
    # ------------------------------------------------------------------

    def _get_sho_params_from_soln(self, resolve):
        """Extract (sigma, omega0, quality) for each SHO component from a solution."""
        components = []
        if getattr(self, "fit_shoterm", False):
            sigma = resolve("sigma")
            rho = resolve("rho")
            Q = resolve("Q")
            omega0 = 2 * np.pi / rho
            components.append(("SHOTerm", sigma, omega0, Q))

        if getattr(self, "fit_rotationterm", False):
            sigma_rot = resolve("sigma_rot")
            period = resolve("period")
            Q0 = resolve("Q0")
            dQ = resolve("dQ")
            f = resolve("f")

            Q1 = 0.5 + Q0 + dQ
            Q2 = 0.5 + Q0
            w1 = 4 * np.pi * Q1 / (period * np.sqrt(4 * Q1**2 - 1))
            w2 = 8 * np.pi * Q2 / (period * np.sqrt(4 * Q2**2 - 1))
            sigma1 = sigma_rot / np.sqrt(1 + f)
            sigma2 = sigma_rot * np.sqrt(f / (1 + f))
            components.append(("Rot1", sigma1, w1, Q1))
            components.append(("Rot2", sigma2, w2, Q2))

        return components

    # ------------------------------------------------------------------
    # PSD / ACF drawing (analytical kernel overlay)
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
            resolve = self._resolve_from_soln(soln)
            omega = 2 * np.pi * freq
            psd_scale = 4 * np.pi

            components = self._get_sho_params_from_soln(resolve)

            if getattr(self, "fit_shoterm", False):
                sho_comps = [c for c in components if c[0] == "SHOTerm"]
                sho_psd = sum(psd_scale * _sho_psd(c[1], c[2], c[3], omega) for c in sho_comps)
                ax.loglog(freq, sho_psd, color="C2", lw=2, ls="--", label="SHOTerm (MAP)")

            if getattr(self, "fit_rotationterm", False):
                rot_comps = [c for c in components if c[0].startswith("Rot")]
                rot_psd = sum(psd_scale * _sho_psd(c[1], c[2], c[3], omega) for c in rot_comps)
                ax.loglog(freq, rot_psd, color="C0", lw=2, ls="--", label="RotationTerm (MAP)")

            if getattr(self, "fit_shoterm", False) and getattr(self, "fit_rotationterm", False):
                full_psd = sum(psd_scale * _sho_psd(c[1], c[2], c[3], omega) for c in components)
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
            resolve = self._resolve_from_soln(soln)
            tau = np.linspace(0, max_lag, n_lags)
            components = self._get_sho_params_from_soln(resolve)

            if getattr(self, "fit_shoterm", False):
                sho_comps = [c for c in components if c[0] == "SHOTerm"]
                sho_acf_vals = sum(_sho_acf(c[1], c[2], c[3], tau) for c in sho_comps)
                sho_acf_vals = sho_acf_vals / sho_acf_vals[0] if sho_acf_vals[0] != 0 else sho_acf_vals
                ax.plot(tau, sho_acf_vals, color="C2", lw=2, ls="--", label="SHOTerm (MAP)")

            if getattr(self, "fit_rotationterm", False):
                rot_comps = [c for c in components if c[0].startswith("Rot")]
                rot_acf_vals = sum(_sho_acf(c[1], c[2], c[3], tau) for c in rot_comps)
                rot_acf_vals = rot_acf_vals / rot_acf_vals[0] if rot_acf_vals[0] != 0 else rot_acf_vals
                ax.plot(tau, rot_acf_vals, color="C0", lw=2, ls="--", label="RotationTerm (MAP)")

            if getattr(self, "fit_shoterm", False) and getattr(self, "fit_rotationterm", False):
                full_acf_vals = sum(_sho_acf(c[1], c[2], c[3], tau) for c in components)
                full_acf_vals = full_acf_vals / full_acf_vals[0] if full_acf_vals[0] != 0 else full_acf_vals
                ax.plot(tau, full_acf_vals, color="C3", lw=2, label="Full kernel (MAP)")

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
                components = self._get_sho_params_from_soln(resolve)

                model_acf = sum(_sho_acf(c[1], c[2], c[3], tau_data) for c in components)
                model_acf_norm = model_acf / model_acf[0] if model_acf[0] != 0 else model_acf
                acf_cost = np.sum((model_acf_norm - acf_data) ** 2)

                model_psd = sum(psd_scale * _sho_psd(c[1], c[2], c[3], omega) for c in components)
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
