import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import h5py
import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.download import download_tess_data
from src.masking import process_all_sectors
import arviz as az


CATALOG = "tess_sp_eb_parameters.csv"
REFIT_CATALOG = "tess_sp_eb_parameters_refit.csv"
HARD_DRIVE_PATH = "/media/disk/tess_eb_data"
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries
REFIT = True
PLOT_PHASE = True
PLOT_DIR = os.path.join(HARD_DRIVE_PATH, "phase_plots") if PLOT_PHASE else None


def process_tic(TIC_ID):
    """Download and mask data for a single TIC ID. Returns None on success, error string on failure."""
    RAW_H5 = os.path.join(HARD_DRIVE_PATH, "raw_data", f"TIC{TIC_ID}.h5")
    MASKED_H5 = os.path.join(HARD_DRIVE_PATH, "masked_data", f"TIC{TIC_ID}.h5")

    # Download raw lightcurves (skip if already downloaded)
    if os.path.exists(RAW_H5):
        print(f"Raw data already exists at {RAW_H5}, skipping download.")
    else:
        print(f"Downloading TESS data for TIC {TIC_ID}...")
        download_tess_data(TIC_ID, download_dir=os.path.join(HARD_DRIVE_PATH, "raw_data"))
        print("Download complete.")

    # Mask transits (skip if already masked)
    if os.path.exists(MASKED_H5) and not REFIT:
        print(f"Masked data already exists at {MASKED_H5}, skipping masking.")
    else:
        print(f"Applying transit masking for TIC {TIC_ID}... {'(will refit transit model)' if REFIT else ''}")
        process_all_sectors(TIC_ID, RAW_H5, catalog_path=CATALOG, output_dir=os.path.dirname(MASKED_H5),
                            mask_factor=1.0, sigma_clip=3, remove_outliers=False,
                            refit=REFIT, refit_catalog_path=REFIT_CATALOG,
                            plot=PLOT_PHASE, plot_dir=PLOT_DIR)
        print("Masking complete.")


def process_with_retries(TIC_ID, max_retries=MAX_RETRIES):
    """Try processing a TIC ID with retries on transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            process_tic(TIC_ID)
            return True
        except ValueError as e:
            # Data/parameter errors won't resolve with retries
            print(f"[FAILED] TIC {TIC_ID}: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] TIC {TIC_ID} attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[FAILED] TIC {TIC_ID} after {max_retries} attempts")
                traceback.print_exc()
    return False


# Determine which IDs to process
if "--retry" in sys.argv or os.path.exists("failed_ids.txt") and len(sys.argv) > 1 and sys.argv[1] == "--retry":
    tic_ids = np.loadtxt("failed_ids.txt", dtype=str).tolist()
    if isinstance(tic_ids, str):
        tic_ids = [tic_ids]
    print(f"Retrying {len(tic_ids)} failed TIC IDs from failed_ids.txt")
else:
    cat = pd.read_csv(CATALOG)
    tic_ids = cat["TIC"].astype(str).tolist()
    print(f"Loaded catalog with {len(tic_ids)} TIC IDs.")

failed_ids = []
for TIC_ID in tqdm.tqdm(tic_ids, desc="Processing TIC IDs"):
    success = process_with_retries(TIC_ID)
    if not success:
        failed_ids.append(TIC_ID)

if failed_ids:
    np.savetxt("failed_ids.txt", failed_ids, fmt="%s")
    print(f"\n{len(failed_ids)} IDs still failed. Saved to failed_ids.txt")
else:
    # Clear the file if all succeeded
    if os.path.exists("failed_ids.txt"):
        os.remove("failed_ids.txt")
    print("\nAll IDs processed successfully!")