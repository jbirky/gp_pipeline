import os
from pathlib import Path

PROJECT_PATH = str(Path.cwd()).strip("scripts")
HARD_DRIVE_PATH = "/media/disk/eb_rotation_gp/"
paths = {
    "SOURCE_PATH": PROJECT_PATH + "source/",
    "CATALOG_PATH": PROJECT_PATH + "catalogs/",
    "DATA_PATH": HARD_DRIVE_PATH + "data/",
    "RESULTS_PATH": HARD_DRIVE_PATH + "results/",
    "PLOTS_PATH": HARD_DRIVE_PATH + "plots/",
}

data_paths = {
    "RAW_LC_PATH": paths["DATA_PATH"] + "raw_lightcurves/",
    "SECTOR_LC_PATH": paths["DATA_PATH"] + "sector_lightcurves/",
    "STITCH_LC_PATH": paths["DATA_PATH"] + "stitched_lightcurves/",
    "MASK_LC_PATH": paths["DATA_PATH"] + "masked_lightcurves/",
    "SMOOTHED_LC_PATH": paths["DATA_PATH"] + "smoothed_lightcurves/",
}
paths.update(data_paths)

plot_paths = {
    "TRANSIT_FIT_PLOTS": paths["PLOTS_PATH"] + "transit_fit_plots/",
    "SMOOTHED_PLOTS": paths["PLOTS_PATH"] + "smoothed_plots/",
    "GP_MAP_PLOTS": paths["PLOTS_PATH"] + "gp_map_plots/",
    "CORNER_PLOTS": paths["PLOTS_PATH"] + "corner_plots/",
}
paths.update(plot_paths)

results_paths = {
    "MAP_RESULTS": paths["RESULTS_PATH"] + "map_results/",
    "POSTERIOR_RESULTS": paths["RESULTS_PATH"] + "posterior_results/",
    "SUMMARY_STATS": paths["RESULTS_PATH"] + "summary_stats/",
}
paths.update(results_paths)
