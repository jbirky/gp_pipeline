import os
import re
import numpy as np
from scipy.optimize import root
from scipy.special import gammaincc

__all__ = ["estimate_inverse_gamma_parameters", "combine_plots_to_pdf"]

def estimate_inverse_gamma_parameters(
    lower, upper, target=0.01, initial=None, **kwargs
):
    r"""Estimate an inverse Gamma with desired tail probabilities
    This method numerically solves for the parameters of an inverse Gamma
    distribution where the tails have a given probability. In other words
    :math:`P(x < \mathrm{lower}) = \mathrm{target}` and similarly for the
    upper bound. More information can be found in `part 4 of this blog post
    <https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html>`_.
    Args:
        lower (float): The location of the lower tail
        upper (float): The location of the upper tail
        target (float, optional): The desired tail probability
        initial (ndarray, optional): An initial guess for the parameters
            ``alpha`` and ``beta``
    Raises:
        RuntimeError: If the solver does not converge.
    Returns:
        dict: A dictionary with the keys ``alpha`` and ``beta`` for the
        parameters of the distribution.
    """
    lower, upper = np.sort([lower, upper])
    if initial is None:
        initial = np.array([2.0, 0.5 * (lower + upper)])
    if np.shape(initial) != (2,) or np.any(np.asarray(initial) <= 0.0):
        raise ValueError("invalid initial guess")

    def obj(x):
        a, b = np.exp(x)
        return np.array(
            [
                gammaincc(a, b / lower) - target,
                1 - gammaincc(a, b / upper) - target,
            ]
        )

    result = root(obj, np.log(initial), method="hybr", **kwargs)
    if not result.success:
        raise RuntimeError(
            "failed to find parameter estimates: \n{0}".format(result.message)
        )
    return dict(zip(("alpha", "beta"), np.exp(result.x)))


# Plot type order: earlier entries appear first within each sector's pages.
_PLOT_TYPE_ORDER = [
    "raw",
    "phase",
    "map_fit",
    "priors",
    "gp_fit",
    "corner",
    "gp_summary",
]


def combine_plots_to_pdf(tic_id, output_dir="output", pdf_path=None):
    """Collect all PNG plots for a TIC target and write them to a single PDF.

    Plots are gathered from every ``<output_dir>/<plot_type>/TIC{tic_id}/``
    subdirectory, grouped by sector (ascending), and ordered within each sector
    by plot type (raw → phase → map_fit → priors → gp_fit → corner → gp_summary).

    Parameters
    ----------
    tic_id : int or str
        TIC identification number.
    output_dir : str
        Base output directory that contains the per-type plot subdirectories.
    pdf_path : str, optional
        Destination path for the PDF. Defaults to
        ``<output_dir>/TIC{tic_id}_plots.pdf``.

    Returns
    -------
    pdf_path : str
        Path to the written PDF file.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages

    tic_id = str(tic_id)
    if pdf_path is None:
        pdf_path = os.path.join(output_dir, f"TIC{tic_id}_plots.pdf")

    # Collect all PNGs that belong to this TIC across all subdirectories.
    sector_re = re.compile(rf"TIC{tic_id}_sector(\d+)_(.+)\.png$")
    found = {}  # (sector, plot_type) -> path

    for entry in os.scandir(output_dir):
        if not entry.is_dir():
            continue
        tic_subdir = os.path.join(entry.path, f"TIC{tic_id}")
        if not os.path.isdir(tic_subdir):
            continue
        for fname in os.listdir(tic_subdir):
            m = sector_re.match(fname)
            if m:
                sector = int(m.group(1))
                plot_type = m.group(2)
                found[(sector, plot_type)] = os.path.join(tic_subdir, fname)

    if not found:
        raise FileNotFoundError(
            f"No PNG plots found for TIC {tic_id} under {output_dir!r}"
        )

    def _sort_key(item):
        (sector, plot_type), _ = item
        type_rank = _PLOT_TYPE_ORDER.index(plot_type) if plot_type in _PLOT_TYPE_ORDER else len(_PLOT_TYPE_ORDER)
        return (type_rank, plot_type, sector)

    ordered = sorted(found.items(), key=_sort_key)

    with PdfPages(pdf_path) as pdf:
        for (sector, plot_type), img_path in ordered:
            img = mpimg.imread(img_path)
            h, w = img.shape[:2]
            fig, ax = plt.subplots(figsize=(w / 100, h / 100))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {len(ordered)} plots to {pdf_path}")
    return pdf_path