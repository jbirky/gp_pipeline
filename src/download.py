import os
import h5py
import numpy as np
from lightkurve import search_lightcurve

from . import config

paths = config.paths


def download_tess_data(tic_id, download_dir=None):
    """Download all available TESS SPOC lightcurves for a TIC ID and save to HDF5.

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    download_dir : str, optional
        Directory for lightkurve cache. If None, uses paths['RAW_LC_PATH'].

    Returns
    -------
    str
        Path to the output HDF5 file.
    """
    if download_dir is None:
        download_dir = paths["RAW_LC_PATH"]

    os.makedirs(download_dir, exist_ok=True)

    tic_id = str(tic_id)
    search = search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")

    if len(search) == 0:
        raise ValueError(f"No TESS SPOC lightcurves found for TIC {tic_id}")

    lc_collection = search.download_all(download_dir=download_dir)

    h5_path = os.path.join(download_dir, f"TIC{tic_id}.h5")

    with h5py.File(h5_path, "w") as f:
        f.attrs["tic_id"] = tic_id
        f.attrs["mission"] = "TESS"
        f.attrs["num_sectors"] = len(lc_collection)

        for lc in lc_collection:
            sector = int(lc.sector)
            grp = f.create_group(f"sector_{sector}")
            grp.attrs["sector"] = sector

            time = lc.time.jd
            flux = lc.flux.value
            flux_err = lc.flux_err.value

            # Normalize flux by sector median
            med_flux = np.nanmedian(flux)
            flux = flux / med_flux
            flux_err = flux_err / med_flux

            grp.create_dataset("time", data=time)
            grp.create_dataset("flux", data=flux)
            grp.create_dataset("flux_err", data=flux_err)

    print(f"Saved {len(lc_collection)} sectors to {h5_path}")
    return h5_path


def list_sectors(h5_path):
    """List available sectors in an HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.

    Returns
    -------
    list of int
        Sorted list of sector numbers.
    """
    sectors = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if key.startswith("sector_"):
                sectors.append(int(key.split("_")[1]))
    return sorted(sectors)
