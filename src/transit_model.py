import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d

import lightkurve
import astropy.units as u
from astropy.timeseries import BoxLeastSquares

__all__ = ["TransitModel",
           "estimate_phase"]


def estimate_phase(time, flux, flux_err, period=None, epoch_time=None, n_bins=100):
    """Phase fold a lightcurve, median bin, and estimate transit phase.

    Parameters
    ----------
    time : array-like
        Time values of the lightcurve (in days).
    flux : array-like
        Flux values of the lightcurve (relative flux).
    flux_err : array-like
        Flux error values of the lightcurve.
    period : float, optional
        Orbital period in days. Defaults to self.sol[6] if sol is set.
    epoch_time : float, optional
        Reference epoch time (t0) in days used to align phase zero.
        If None, lightkurve defaults to the first time value.
    n_bins : int, optional
        Number of equal-width bins for median binning. Default is 100.

    Returns
    -------
    lc_folded : lightkurve.FoldedLightCurve
        Phase-folded lightcurve (unbinned).
    lc_binned : lightkurve.LightCurve
        Median-binned phase-folded lightcurve.
    phase_min : float
        Phase value (in units of period) at minimum flux, estimated as
        the minimum of the 1D cubic interpolation of the binned data.
    interp_func : scipy.interpolate.interp1d
        1D interpolation function fitted to the binned (phase, flux) data.
    """
    lc = lightkurve.LightCurve(time=time, flux=flux, flux_err=flux_err)

    lc_folded = lc.fold(period=period, epoch_time=epoch_time)
    lc_binned = lc_folded.bin(bins=n_bins, aggregate_func=np.nanmedian)

    phase_binned = lc_binned.phase.value
    flux_binned = lc_binned.flux.value

    valid = np.isfinite(phase_binned) & np.isfinite(flux_binned)
    phase_valid = phase_binned[valid]
    flux_valid = flux_binned[valid]

    # Fit 1D cubic interpolation to binned data; fall back to linear if
    # there are fewer than 4 valid points (cubic requires >=4).
    kind = "cubic" if valid.sum() >= 4 else "linear"
    interp_func = interp1d(phase_valid, flux_valid, kind=kind, fill_value="extrapolate")

    # Find the minimum of the interpolated function within the valid range.
    x0 = np.array([phase_valid[flux_valid.argmin()]])  # start near the minimum binned flux
    result = minimize(interp_func, x0=x0, bounds=[(phase_valid.min(), phase_valid.max())])
    phase_min = float(result.x[0])

    return lc_folded, lc_binned, phase_min, interp_func
    

class TransitModel(object):

    def __init__(self,
                 ID,
                 lc_raw=None,
                 mission=None,
                 pipeline=None,
                 prot=None,
                 sol=None):

        self.ID = str(ID)
        if (mission == 'Kepler') or (mission == 'K2'):
            mission = 'KIC'
        if (mission == 'TESS'):
            mission = 'TIC'

        self.pipeline = pipeline
        self.lc_raw = lc_raw
        self.prot = prot
        self.sol = sol

        self.pname = [r"$A_1$", r"$t_1$", r"$\sigma_1$", r"$A_2$", r"$t_2$", r"$\sigma_2$", r"$P_{orb}$"]


    def get_arrays(self, lc, downsample=1):
        x = lc.time.value[::downsample]
        y = lc.flux.value[::downsample]
        s = lc.flux_err.value[::downsample]
        return x, y, s


    def est_duration(self, sol=None, bound=1e-3):
        if sol is None:
            sol = self.sol

        perc = [bound, 1-bound]

        dmin1 = norm.ppf(perc[0], loc=sol[1], scale=sol[2])
        dmax1 = norm.ppf(perc[1], loc=sol[1], scale=sol[2])
        duration1 = dmax1 - dmin1

        dmin2 = norm.ppf(perc[0], loc=sol[4], scale=sol[5])
        dmax2 = norm.ppf(perc[1], loc=sol[4], scale=sol[5])
        duration2 = dmax2 - dmin2

        self.dur1 = duration1
        self.dur2 = duration2

        return self.dur1, self.dur2


    def apply_transit_mask(self, sol=None, mask_factor=1., sigma_clip=3, remove_outliers=False):
        if sol is None:
            sol = self.sol

        porb = sol[6]
        dur1, dur2 = self.est_duration(sol)
        x, y, s = self.get_arrays(self.lc_raw)

        # Check for NaN parameters — cannot mask without valid transit params
        if np.isnan(porb) or np.isnan(dur1) or np.isnan(dur2):
            raise ValueError(f"Transit parameters contain NaN values (Porb={porb}, dur1={dur1}, dur2={dur2})")

        bls = BoxLeastSquares(x*u.day, y, dy=s)

        dur = mask_factor * max(dur1, dur2)
        # Clamp duration to be less than the orbital period (BLS requirement)
        if dur >= porb:
            dur = 0.95 * porb
        tmask1 = bls.transit_mask(x, porb, dur, min(x)+sol[1])
        tmask2 = bls.transit_mask(x, porb, dur, min(x)+sol[4])
        tmask = tmask1 | tmask2
        rmask = ~tmask

        self.tmask = tmask
        self.rmask = rmask

        time, flux, err = self.get_arrays(self.lc_raw)

        t_rmask = time[self.rmask]
        f_rmask = flux[self.rmask]
        e_rmask = err[self.rmask]

        t_tmask = time[self.tmask]
        f_tmask = flux[self.tmask]
        e_tmask = err[self.tmask]

        self.lc_rmask_array = [t_rmask, f_rmask, e_rmask]
        self.lc_tmask_array = [t_tmask, f_tmask, e_tmask]

        self.lc_rmask = lightkurve.LightCurve(time=t_rmask, flux=f_rmask, flux_err=e_rmask)
        self.lc_tmask = lightkurve.LightCurve(time=t_tmask, flux=f_tmask, flux_err=e_tmask)

        if remove_outliers:
            self.lc_rmask = self.lc_rmask.remove_outliers(sigma=sigma_clip)
            self.lc_tmask = self.lc_tmask.remove_outliers(sigma=sigma_clip)

        return self.tmask, self.rmask

    def get_folded(self, porb):
        """
        Returns folded lightcurve arrays.
        """
        self.fold_period = porb
        self.lc_fold = self.lc_flat.fold(period=porb)

        x, y, s = self.get_arrays(self.lc_fold)

        return x, y, s


    def model(self, theta):
        """
        Model of eclipsing binary lightcurves.
        """
        
        a1, t1, d1, a2, t2, d2, porb = theta

        x, y, s = self.get_folded(porb)

        # Double gaussian transit model
        model_lc = np.ones(len(x))

        offsets = [-porb, 0, porb]
        for tc in offsets:
            model_lc -= a1 * np.exp(-0.5 * (x - t1 + tc)**2 / d1**2) 
            model_lc -= a2 * np.exp(-0.5 * (x - t2 + tc)**2 / d2**2) 
        
        return model_lc, y, s
    
    
    def chisq(self, theta, porb_bounds):
        """
        Calculates chi-squared of model fit.
        """
        a1, t1, d1, a2, t2, d2, porb = theta
        
        if (porb < porb_bounds[0]) or (porb > porb_bounds[1]):
            return np.inf
        
        # req positive stdev
        if (d1 <= 0) or (d2 <= 0):
            return np.inf
        elif (d1 >= porb/2) or (d2 >= porb/2):
            return np.inf
        
        # req postive amplitude
        if (a1 <= 0) or (a2 <= 0):
            return np.inf
        
        # req secondary not w/in duration of primary
        if (t2 < t1) and (t2+d2 > t1-d1):
            return np.inf
        if (t1 < t2) and (t1+d1 > t2-d2):
            return np.inf

        # req transit fit to be between +/-porb/2
        if not (-porb/2 < t1 < porb/2):
            return np.inf
        elif not  (-porb/2 < t2 < porb/2):
            return np.inf
        
        # chi^2 between model and data
        ypred, y, s = self.model(theta)
        chi2 = 0.5 * np.nansum((y - ypred)**2/s**2)
        
        return chi2
    
    
    def refit_model(self, t0=None, method='nelder-mead', porb_bounds=None):
        
        # initial guess 
        if t0 is None:
            t0 = self.sol

        # optimize model fit
        res = minimize(self.chisq, t0, method=method, args=(porb_bounds,))
        self.res = res

        self.sol = res.x
        self.chi_fit = res.fun

        return self.sol, self.chi_fit
    
    
    def chisq_period_and_offset(self, tvar, porb_bounds, tfixed):
        """
        Calculates chi-squared of model fit, only varying period.
        """

        porb, toffset = tvar

        theta = np.zeros(7)
        theta[:6] = tfixed[:6]
        theta[6] = porb

        # apply timing offset 
        theta[1] += toffset
        theta[4] += toffset
        
        if (porb < porb_bounds[0]) or (porb > porb_bounds[1]):
            return np.inf
        
        # chi^2 between model and data
        ypred, y, s = self.model(theta)
        chi2 = 0.5 * np.nansum((y - ypred)**2/s**2)
        
        return chi2
    
    
    def refit_model_period_and_offset(self, porb0, toffset0, tfixed, method='nelder-mead', porb_bounds=None, n_toffset_grid=10):

        if porb_bounds is None:
            porb_bounds = [.5*porb0, 1.5*porb0]

        toffset_step = tfixed[2]  # scale step by d1 (transit sigma)

        # Build a grid of initial toffset values spanning +-porb/2
        toffset_grid = np.linspace(-porb0 / 2, porb0 / 2, n_toffset_grid)

        best_res = None
        best_chi = np.inf
        best_toffset_init = toffset_grid[0]

        for toffset_init in toffset_grid:
            tvar0 = np.array([porb0, toffset_init])
            initial_simplex = np.array([
                tvar0,
                tvar0 + [porb0 * 0.05, 0.0],
                tvar0 + [0.0, toffset_step],
            ])
            res = minimize(self.chisq_period_and_offset, tvar0, method=method,
                           options={'initial_simplex': initial_simplex},
                           args=(porb_bounds, tfixed))
            if res.fun < best_chi:
                best_chi = res.fun
                best_res = res
                best_toffset_init = toffset_init

        self.res = best_res
        porb, toffset = best_res.x

        print(f"Nopt grid size: {n_toffset_grid}, best toffset_init: {best_toffset_init:.5f} days")
        print(f"Nopt iterations: {best_res.nit}")
        print(f"Initial porb: {porb0:.5f} days, toffset: {toffset0:.5f} days")
        print(f"Refined porb: {porb:.5f} days, toffset: {toffset:.5f} days")
        print("")

        # apply timing offset and update Porb in-place
        tfixed[1] += toffset
        tfixed[4] += toffset
        tfixed[6] = porb

        self.sol = np.array(tfixed)
        self.chi_fit = best_res.fun

        return self.sol, self.chi_fit
