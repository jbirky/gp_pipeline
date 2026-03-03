import os
import h5py
import numpy as np
import pandas as pd
import lightkurve

from .transit_model import TransitModel, estimate_phase
from .download import list_sectors
from . import config

paths = config.paths

PARAM_NAMES = ["a1", "t1", "d1", "a2", "t2", "d2", "Porb"]


def load_sector_from_h5(h5_path, sector):
    """Load a single sector's data from an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the raw data HDF5 file.
    sector : int
        Sector number to load.

    Returns
    -------
    time, flux, flux_err : np.ndarray
        Arrays for the requested sector.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[f"sector_{sector}"]
        time = grp["time"][:]
        flux = grp["flux"][:]
        flux_err = grp["flux_err"][:]
    return time, flux, flux_err


def get_object_params(tic_id, catalog_path=None, sector=None, refit_catalog_path=None):
    """Read transit parameters for a TIC ID from the EB catalog.

    If ``refit_catalog_path`` and ``sector`` are both provided, the refit
    catalog is checked first for a (TIC, sector) match and its values are
    returned when found.  Otherwise the original catalog is used.

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    catalog_path : str, optional
        Path to the CSV catalog. Defaults to tess_sp_eb_parameters.csv in
        the project root.
    sector : int, optional
        Sector number. When provided together with ``refit_catalog_path``,
        enables per-sector lookup in the refit catalog.
    refit_catalog_path : str, optional
        Path to the refit catalog CSV. When a matching (TIC, sector) row
        exists, those values take priority over the original catalog.

    Returns
    -------
    dict
        Dictionary with keys: a1, t1, d1, a2, t2, d2, Porb, Prot.
    """
    # Check refit catalog first when sector is specified
    if sector is not None and refit_catalog_path and os.path.exists(refit_catalog_path):
        df_refit = pd.read_csv(refit_catalog_path)
        match = df_refit[(df_refit["TIC"] == int(tic_id)) & (df_refit["sector"] == sector)]
        if len(match) > 0:
            row = match.iloc[0]
            params = {p: row[p] for p in PARAM_NAMES}
            params["Prot"] = row["Prot"]
            return params

    if catalog_path is None:
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tess_sp_eb_parameters.csv",
        )

    sp = pd.read_csv(catalog_path)

    # If the catalog has a "sector" column (i.e. it is the refit catalog used
    # without a sector key), average Prot and take the first row's transit params.
    if "sector" in sp.columns:
        rows = sp[sp["TIC"] == int(tic_id)]
        if len(rows) == 0:
            raise ValueError(f"TIC {tic_id} not found in catalog {catalog_path}")
        params = {p: rows[p].iloc[0] for p in PARAM_NAMES}
        params["Prot"] = rows["Prot"].mean()
        return params

    row = sp[sp["TIC"] == int(tic_id)]
    if len(row) == 0:
        raise ValueError(f"TIC {tic_id} not found in catalog {catalog_path}")

    params = {p: row[p].iloc[0] for p in PARAM_NAMES}
    params["Prot"] = row["Prot"].iloc[0]
    return params


def save_refit_params(tic_id, sector, sol, prot, chi_fit, refit_catalog_path,
                      phase_orb=None, phase_rot=None):
    """Save refitted transit parameters for one sector to the refit catalog.

    Creates the file if it does not exist.  Replaces any existing row for
    the same (TIC, sector) pair.

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    sector : int
        Sector number.
    sol : list or array of length 7
        Refitted parameters [a1, t1, d1, a2, t2, d2, Porb].
    prot : float
        Rotation period (unchanged by transit refit; carried over from the
        original catalog).
    chi_fit : float
        Chi-squared value of the refit.
    refit_catalog_path : str
        Path to the refit catalog CSV.
    phase_orb : float, optional
        Phase at minimum flux of the transit lightcurve folded at Porb.
    phase_rot : float, optional
        Phase at minimum flux of the rotation lightcurve folded at Prot.
    """
    row = {
        "TIC": int(tic_id),
        "sector": int(sector),
        "a1": sol[0], "t1": sol[1], "d1": sol[2],
        "a2": sol[3], "t2": sol[4], "d2": sol[5],
        "Porb": sol[6],
        "Prot": prot,
        "chi_fit": chi_fit,
        "phase_orb": phase_orb,
        "phase_rot": phase_rot,
    }

    if os.path.exists(refit_catalog_path):
        df = pd.read_csv(refit_catalog_path)
        df = df[~((df["TIC"] == int(tic_id)) & (df["sector"] == int(sector)))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(refit_catalog_path, index=False)
    print(f"Saved refit params for TIC {tic_id} sector {sector} to {refit_catalog_path}")


def _plot_phase_diagnostics(tic_id, sector, period,
                            lc_folded_orb, lc_binned_orb, phase_min_orb, interp_orb,
                            lc_folded_rot, lc_binned_rot, phase_min_rot, interp_rot,
                            plot_dir):
    """Save a two-panel phase diagnostic figure for one sector.

    Left panel shows the transit (orbital) phased data; right panel shows the
    rotation phased data.  Each panel contains the raw phased scatter, the
    median-binned points, the 1D interpolation curve, and a vertical marker at
    the interpolated flux minimum.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)
    fig.suptitle(f"TIC {tic_id}  Sector {sector}  |  Period = {period:.4f} d")

    datasets = [
        (axes[0], lc_folded_orb, lc_binned_orb, phase_min_orb, interp_orb, "Transit", "red"),
        # (axes[1], lc_folded_orb, lc_binned_orb, phase_min_orb, interp_orb, "Transit", "red"),
        (axes[1], lc_folded_rot, lc_binned_rot, phase_min_rot, interp_rot, "Rotation", "blue"),
    ]

    for ax, lc_folded, lc_binned, phase_min, interp_func, title, color in datasets:
        if lc_folded is None:
            ax.set_title(f"{title} (unavailable)")
            continue

        phase_f = lc_folded.phase.value
        flux_f = lc_folded.flux.value
        phase_b = lc_binned.phase.value
        flux_b = lc_binned.flux.value

        ax.scatter(phase_f, flux_f, s=1, alpha=0.2, color="gray", label="Data")
        ax.scatter(phase_b, flux_b, s=15, color=color, zorder=5, label="Binned")

        if interp_func is not None:
            valid = np.isfinite(phase_b) & np.isfinite(flux_b)
            phase_dense = np.linspace(phase_b[valid].min(), phase_b[valid].max(), 500)
            ax.plot(phase_dense, interp_func(phase_dense), color=color, lw=1.5, label="Interpolation")

        if phase_min is not None:
            ax.axvline(phase_min, color="purple", lw=1.5, ls="--",
                       label=f"Min phase: {phase_min:.4f}")

        ax.set_ylabel("Flux", fontsize=22)
        ax.set_title(title)
        ax.legend(fontsize=12)
        
        yspan = np.nanmax(flux_b) - np.nanmin(flux_b)
        ax.set_ylim([np.nanmin(flux_b) - 0.1*yspan, np.nanmax(flux_b) + 0.1*yspan])

    axes[1].set_xlabel("Phase", fontsize=22)

    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"TIC{tic_id}_sector{sector}_phase.png")
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved phase diagnostic plot to {plot_path}")


def process_sector(tic_id, sector, h5_path, catalog_path=None, output_dir=None,
                   mask_factor=1.0, sigma_clip=3, remove_outliers=False,
                   refit_offset=False, refit_all=False, refit_catalog_path=None,
                   plot=False, plot_dir=None):
    """Apply transit masking to a single sector and save results.

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    sector : int
        Sector number to process.
    h5_path : str
        Path to the raw data HDF5 file.
    catalog_path : str, optional
        Path to the EB parameter catalog CSV.
    output_dir : str, optional
        Directory for output HDF5. Defaults to paths['MASK_LC_PATH'].
    refit : bool, optional
        Whether to refit the transit model. Defaults to False.
    refit_catalog_path : str, optional
        Path to the refit catalog CSV. When provided, updated parameters and
        phase estimates are written here after each sector is processed. On
        subsequent runs the refit catalog is used as the starting point for
        this (TIC, sector).
    plot : bool, optional
        When True, save a two-panel phase diagnostic figure (transit and
        rotation phased data with interpolation and min-phase marker).
    plot_dir : str, optional
        Directory for phase diagnostic figures. Required when ``plot=True``.

    Returns
    -------
    str
        Path to the output masked HDF5 file.
    """
    if output_dir is None:
        output_dir = paths["MASK_LC_PATH"]
    os.makedirs(output_dir, exist_ok=True)

    tic_id = str(tic_id)
    time, flux, flux_err = load_sector_from_h5(h5_path, sector)

    params = get_object_params(tic_id, catalog_path, sector=sector,
                               refit_catalog_path=refit_catalog_path)
    sol = [params[p] for p in PARAM_NAMES]

    lc_raw = lightkurve.LightCurve(time=time, flux=flux, flux_err=flux_err)

    ebm = TransitModel(tic_id, lc_raw=lc_raw, mission="TESS", prot=params["Prot"])
    ebm.lc_flat = lc_raw
    if refit_offset:
        print(f"Refitting transit model [porb, offset] for TIC {tic_id} sector {sector}...")
        ebm.refit_model_period_and_offset(params["Porb"], 0, sol, method='nelder-mead', porb_bounds=(0.95*params["Porb"], 1.05*params["Porb"]))
    if refit_all:
        print(f"Refitting transit model [all parameters] for TIC {tic_id} sector {sector}...")
        ebm.refit_model(t0=sol, method='nelder-mead', porb_bounds=(0.95*params["Porb"], 1.05*params["Porb"]))
    ebm.apply_transit_mask(sol=sol, mask_factor=mask_factor, sigma_clip=sigma_clip, remove_outliers=remove_outliers)

    # Estimate orbital phase from transit-masked data and rotation phase from
    # rotation-masked data.  Run whenever the results are needed for saving
    # to the refit catalog or for generating a diagnostic plot.
    phase_orb = phase_rot = None
    lc_folded_orb = lc_binned_orb = interp_orb = None
    lc_folded_rot = lc_binned_rot = interp_rot = None

    if refit_catalog_path is not None or plot:
        epoch_time = min(min(ebm.lc_rmask.time.value), min(ebm.lc_tmask.time.value))

        try:
            xt = ebm.lc_tmask.time.value
            yt = ebm.lc_tmask.flux.value
            et = ebm.lc_tmask.flux_err.value
            lc_folded_orb, lc_binned_orb, phase_orb, interp_orb = estimate_phase(
                xt, yt, et, period=sol[6], epoch_time=epoch_time)
        except Exception as e:
            print(f"  Warning: could not estimate orbital phase for TIC {tic_id} sector {sector}: {e}")

        # note: rotation phase is in terms of the orbital period!
        try:
            xr = ebm.lc_rmask.time.value
            yr = ebm.lc_rmask.flux.value
            er = ebm.lc_rmask.flux_err.value
            lc_folded_rot, lc_binned_rot, phase_rot, interp_rot = estimate_phase(
                xr, yr, er, period=sol[6], epoch_time=epoch_time)
        except Exception as e:
            print(f"  Warning: could not estimate rotation phase for TIC {tic_id} sector {sector}: {e}")

    if refit_catalog_path is not None:
        save_refit_params(tic_id, sector, sol, params["Prot"], ebm.chi_fit,
                          refit_catalog_path, phase_orb=phase_orb, phase_rot=phase_rot)

    if plot and plot_dir is not None:
        _plot_phase_diagnostics(
            tic_id, sector, sol[6],
            lc_folded_orb, lc_binned_orb, phase_orb, interp_orb,
            lc_folded_rot, lc_binned_rot, phase_rot, interp_rot,
            plot_dir,
        )

    out_path = os.path.join(output_dir, f"TIC{tic_id}.h5")

    with h5py.File(out_path, "a") as f:
        # Transit-masked data (the in-transit points)
        grp_name = f"transit_masked/sector{sector}"
        if grp_name in f:
            del f[grp_name]
        grp = f.create_group(grp_name)
        grp.create_dataset("time", data=np.array(ebm.lc_tmask.time.value))
        grp.create_dataset("flux", data=np.array(ebm.lc_tmask.flux.value))
        grp.create_dataset("flux_err", data=np.array(ebm.lc_tmask.flux_err.value))

        # Rotation-masked data (transits removed, rotation signal preserved)
        grp_name = f"rotation_masked/sector{sector}"
        if grp_name in f:
            del f[grp_name]
        grp = f.create_group(grp_name)
        grp.create_dataset("time", data=np.array(ebm.lc_rmask.time.value))
        grp.create_dataset("flux", data=np.array(ebm.lc_rmask.flux.value))
        grp.create_dataset("flux_err", data=np.array(ebm.lc_rmask.flux_err.value))

    print(f"Saved masked sector {sector} to {out_path}")
    return out_path


def process_all_sectors(tic_id, h5_path, catalog_path=None, output_dir=None,
                        mask_factor=1.0, sigma_clip=3, remove_outliers=False,
                        refit_offset=False, refit_all=False, refit_catalog_path=None,
                        plot=False, plot_dir=None):
    """Apply transit masking to all sectors for a given TIC ID.

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    h5_path : str
        Path to the raw data HDF5 file.
    catalog_path : str, optional
        Path to the EB parameter catalog CSV.
    output_dir : str, optional
        Directory for output HDF5.
    refit_catalog_path : str, optional
        Path to the refit catalog CSV. Passed through to ``process_sector``.
    plot : bool, optional
        When True, save per-sector phase diagnostic figures.
    plot_dir : str, optional
        Directory for phase diagnostic figures.

    Returns
    -------
    str
        Path to the output masked HDF5 file.
    """
    sectors = list_sectors(h5_path)
    out_path = None
    for sector in sectors:
        print("")
        out_path = process_sector(
            tic_id, sector, h5_path, catalog_path, output_dir,
            mask_factor, sigma_clip, remove_outliers, refit_offset, refit_all, refit_catalog_path,
            plot=plot, plot_dir=plot_dir,
        )
    print(f"Processed {len(sectors)} sectors for TIC {tic_id}")
    return out_path
