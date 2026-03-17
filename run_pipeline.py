import os
os.environ["JAX_ENABLE_X64"] = "True"
# os.environ["JAX_PLATFORMS"] = "gpu"

import numpy as np
import pandas as pd
import h5py
import warnings
warnings.filterwarnings("ignore")

from src.download import download_tess_data
from src.masking import process_all_sectors
from src.gp_fit import TessObject, GPFit, WalkerPlotCallback
import arviz as az

# ===================================================================
# Set up paths and parameters for the pipeline
# ===================================================================

CATALOG = "catalogs/tess_sp_eb_parameters_no_flags.csv"
REFIT_CATALOG = "catalogs/tess_sp_eb_parameters_no_flags_refit.csv"
TIC_LIST = pd.read_csv(CATALOG)["TIC"].astype(str).tolist()

TIC_ID = "202490797"
# TIC_ID = "233650330"

# -----------------------------------------------
# data preprocessing options

BIN_SIZE = 30. / 24/60   # minutes
DOWNSAMPLE = 1
REFIT_MASK = False
REFIT_MAP = False
MASK_FACTOR = 1.
REMOVE_OUTLIERS = True
PLOT_PHASE = True

# -----------------------------------------------
# MCMC sampling options

NCHAINS = 1
SAMPLER = "pymc"
TUNE = 2500
DRAWS = 2500
TARGET_ACCEPT = 0.9
STEP_SCALE = 0.25
PYMC_CALLBACK = False
USE_MAP_INIT = True

# -----------------------------------------------
# Directories and file paths

OUTPUT_DIR = "results/" + TIC_ID
HARD_DRIVE_PATH = "/mnt/tess_eb_data"
RAW_H5 = os.path.join(HARD_DRIVE_PATH, "raw_data", f"TIC{TIC_ID}.h5")
MASKED_H5 = os.path.join(HARD_DRIVE_PATH, "masked_data", f"TIC{TIC_ID}.h5")
PLOT_DIR = os.path.join(OUTPUT_DIR, f"phase_plots/TIC{TIC_ID}") if PLOT_PHASE else None
WALKER_PLOT_DIR = os.path.join(OUTPUT_DIR, "walker_plots") if PYMC_CALLBACK else None
MAP_OUT_DIR = os.path.join(OUTPUT_DIR, "map_results")
MAP_PLOT_DIR = os.path.join(OUTPUT_DIR, "gp_map_plots")
PSD_PLOT_DIR = os.path.join(OUTPUT_DIR, "gp_psd_plots")
ACF_PLOT_DIR = os.path.join(OUTPUT_DIR, "gp_acf_plots")
LC_PHASE_PLOT_DIR = os.path.join(OUTPUT_DIR, "lc_phase_plots")

for _dir in [PLOT_DIR, WALKER_PLOT_DIR, MAP_OUT_DIR, MAP_PLOT_DIR, PSD_PLOT_DIR, ACF_PLOT_DIR, LC_PHASE_PLOT_DIR]:
    if _dir is not None:
        os.makedirs(_dir, exist_ok=True)

# ===================================================================
# Download, process, and mask TESS data for the target TIC
# ===================================================================

# Get sectors
sectors = []
with h5py.File(MASKED_H5, "r") as f:
    for key in f["rotation_masked"].keys():
        sectors.append(int(key.replace("sector", "")))
sectors.sort()
print(f"Sectors to fit: {sectors}")

# Download raw lightcurves (skip if already downloaded)
if os.path.exists(RAW_H5):
    print(f"Raw data already exists at {RAW_H5}, skipping download.")
else:
    print(f"Downloading TESS data for TIC {TIC_ID}...")
    download_tess_data(TIC_ID, download_dir=os.path.join(HARD_DRIVE_PATH, "raw_data"))
    print("Download complete.")

# Mask transits (skip if already masked)
if os.path.exists(MASKED_H5) and not REFIT_MASK:
    print(f"Masked data already exists at {MASKED_H5}, skipping masking.")
else:
    print(f"Applying transit masking for TIC {TIC_ID}... {'(will refit transit model)' if REFIT_MASK else ''}")
    process_all_sectors(TIC_ID, RAW_H5, catalog_path=CATALOG, output_dir=os.path.dirname(MASKED_H5),
                        mask_factor=1.0, sigma_clip=3, remove_outliers=False,
                        refit_offset=REFIT_MASK, refit_catalog_path=REFIT_CATALOG,
                        plot=PLOT_PHASE, plot_dir=PLOT_DIR)

# ===================================================================
# Load TESS data for the target sector
# ===================================================================

sector = sectors[3]

tess = TessObject(
    TIC_ID,
    masked_h5_path=MASKED_H5,
    output_dir=OUTPUT_DIR,
    catalog_path=REFIT_CATALOG if os.path.exists(REFIT_CATALOG) else CATALOG,
    bin_size=BIN_SIZE,
    downsample=DOWNSAMPLE,
)

x, y, yerr = tess.load_data(sector=sector)

p_guess = tess.prot_init
max_lag = 4 * p_guess
label = f"TIC{TIC_ID}_sector{tess.sector_label}"

# ===================================================================
# Initialize GPFit and priors for this target
# ===================================================================

gp = GPFit(x, y, yerr, label=label, output_dir=OUTPUT_DIR)

prior_params = {
    "mean":          {"type": "fixed",   "value": float(np.mean(y))},
    "log_jitter":    {"type": "normal",  "mu": float(np.log(np.mean(yerr))), "sigma": 2.0},
    # SHOTerm: short-timescale variability (granulation / systematics)
    "log_sigma":     {"type": "normal",  "mu": -1.0, "sigma": 2.0},
    "log_rho":       {"type": "normal",  "mu": float(np.log(np.median(np.diff(x)))), "sigma": 2.0},
    "Q":             {"type": "fixed",   "value": 1./3.},
    # RotationTerm: quasi-periodic variability (starspot rotation)
    "period":        {"type": "normal",  "mu": p_guess, "sigma": 1.0},
    "log_sigma_rot": {"type": "normal",  "mu": float(np.log(np.std(y))), "sigma": 5.0},
    "log_Q0":        {"type": "normal",  "mu": 0.0, "sigma": 5.0},
    "log_dQ":        {"type": "normal",  "mu": 0.0, "sigma": 5.0},
    "f":             {"type": "uniform", "lower": 0.0, "upper": 1.0},
}

if SAMPLER == "pymc":
    gp.build_pymc_model(prior_params=prior_params)
elif SAMPLER == "numpyro":
    gp.build_numpyro_model(prior_params=prior_params)

soln = gp.fit_kernel_initial(max_lag=max_lag, acf_weight=1.0, psd_weight=0.1, n_freq=int(5e5),
                             log_spacing=True)
gp.plot_kernel_components(soln=soln, output_dir=MAP_PLOT_DIR)
gp.plot_phase_fold(soln=soln, output_dir=LC_PHASE_PLOT_DIR,
                   x_transit=tess.x_transit, y_transit=tess.y_transit)
gp.plot_psd_acf(soln=soln, show_kernel=True, n_freq=int(5e5), log_spacing=True,
                output_dir=PSD_PLOT_DIR, plot_suffix="_initial_fit")
gp.plot_priors(soln=soln, return_fig=False, output_dir=MAP_PLOT_DIR)

# breakpoint()
# # Re-center priors on the initial solution, keeping the original prior family
# for key in prior_params.keys():
#     if (key in soln) and (prior_params[key]["type"] != "fixed"):
#         spec = prior_params[key]
#         val = soln[key]
#         if spec["type"] == "normal":
#             prior_params[key] = {"type": "normal", "mu": val, "sigma": spec["sigma"]}
#         elif spec["type"] == "uniform":
#             # Keep original width, shift center to MAP
#             half_width = 0.5 * (spec["upper"] - spec["lower"])
#             prior_params[key] = {"type": "uniform", "lower": val - half_width, "upper": val + half_width}

if SAMPLER == "pymc":
    gp.build_pymc_model(prior_params=prior_params)
elif SAMPLER == "numpyro":
    gp.build_numpyro_model(prior_params=prior_params)


# ===================================================================
# Run MCMC sampling (uses cached MAP + model)
# ===================================================================

# print(f"\n{'='*60}")
# print(f"MCMC sampling TIC {TIC_ID} sector {sector} \n{SAMPLER}: tune={TUNE}, draws={DRAWS}, chains={NCHAINS}, target_accept={TARGET_ACCEPT}, step_scale={STEP_SCALE}")
# print(f"{'='*60}\n")

# free_var_names = gp.free_var_names
# sector_map_soln = {k: float(soln[k]) for k in free_var_names if k in soln}

# print(f"Running MCMC sampling for sector {sector} with sampler {SAMPLER}.")

# walker_plot_path = os.path.join(WALKER_PLOT_DIR, f"{label}_walker_plot.png") if WALKER_PLOT_DIR else None
# callback = WalkerPlotCallback(free_var_names, map_soln=sector_map_soln, save_path=walker_plot_path, ylims=gp.get_walker_ylims(sigma_factor=3.0), priors=gp._get_priors())
# trace = gp.fit(
#     sampler=SAMPLER,
#     tune=TUNE,
#     draws=DRAWS,
#     chains=NCHAINS,
#     cores=NCHAINS,
#     target_accept=TARGET_ACCEPT,
#     map_soln=sector_map_soln if USE_MAP_INIT else {},
#     callback=callback if PYMC_CALLBACK else None,
#     step_scale=STEP_SCALE,
#     init_jitter=0.1
# )
