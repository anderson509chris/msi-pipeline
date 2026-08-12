"""Ion images: per-pixel intensity maps built from pass1's peak_<tag>.npy +
coords_<tag>.pkl. pass1 already computes, per target, the max intensity in
every pixel within +/-PPM of that m/z - this stage is the first thing that
actually renders that as a 2D image of the tissue.

    python ionimage.py "<run_dir>" [--mode prof|cent] [--out DIR] [--targets targets.json]
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import math
import pickle
import textwrap
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from msi_io import RunConfig, RunConfigError, add_run_dir_args

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8})
CMAP = "magma"
BG_COLOR = "#d9d9d9"   # unsampled pixel - visually distinct from any real (low) intensity
ROI_COLOR = "#29b6f6"


def build_grid(coords, values):
    """coords: list of 1-based (x, y) pixel positions, length n. values: an
    array of length n, same order as coords (as pass1 writes them). Returns
    a 2D array of shape (max_y, max_x): grid[y-1, x-1] = values[i].

    Positions with no corresponding entry in coords stay NaN. Unsampled is
    not the same as zero - a pixel that really measured zero intensity is
    valid data, and filling gaps with zero would make it indistinguishable
    from "this position was never acquired"."""
    n = len(coords)
    xs = np.fromiter((c[0] for c in coords), dtype=np.int64, count=n)
    ys = np.fromiter((c[1] for c in coords), dtype=np.int64, count=n)
    max_x = int(xs.max()) if n else 0
    max_y = int(ys.max()) if n else 0
    grid = np.full((max_y, max_x), np.nan, dtype=np.float64)
    grid[ys - 1, xs - 1] = values
    return grid


def compute_vmax(grid, floor, percentile):
    """The percentile (default 99th) of non-NaN, above-floor pixel values -
    deliberately NOT the raw maximum. MSI data routinely has one or two
    extreme pixels (a salt crystal, an edge artifact); scaling to the true
    max flattens everything else in the image into a uniform dark rectangle,
    which is the single most common way an ion image misleads."""
    finite = grid[np.isfinite(grid)]
    above_floor = finite[finite > floor]
    if above_floor.size:
        return float(np.percentile(above_floor, percentile))
    if finite.size:
        return float(finite.max())
    return 1.0


def load_roi_pixels(cfg, roi, T):
    """Pixel coordinates to outline for target T, or None. Returns None
    (with a printed warning, not an exception) if the relevant pickle
    doesn't exist yet, so this stage can be run standalone right after
    pass1 - before pass2/common have produced spectra.pkl / common_pixels.pkl."""
    if roi == "none":
        return None
    if roi == "per-target":
        path = Path(cfg.out("spectra.pkl"))
        if not path.is_file():
            print(f"WARNING: --roi per-target requested but {path.name} not found yet "
                  f"(run pass2.py first) - skipping ROI overlay", flush=True)
            return None
        with open(path, "rb") as f:
            spectra = pickle.load(f)
        return spectra.get(T, {}).get("pixels")
    if roi == "common":
        path = Path(cfg.out("common_pixels.pkl"))
        if not path.is_file():
            print(f"WARNING: --roi common requested but {path.name} not found yet "
                  f"(run common.py first) - skipping ROI overlay", flush=True)
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"unknown --roi value: {roi}")


def target_title(cfg, T):
    """Target name, m/z (4 decimals), and formula when present - matching
    the label content shown elsewhere (plot.py's panels, the CSV's
    "assignment" column)."""
    name = cfg.names.get(T, "")
    formula = cfg.formulas.get(T, "")
    mz_line = f"m/z {T:.4f}"
    has_name = bool(name) and name != str(T)
    if not has_name:
        return mz_line + (f"  ({formula})" if formula else "")
    head = f"{name}  ({formula})" if formula else name
    return f"{head}\n{mz_line}"


def render_panel(ax, grid, vmin, vmax, roi_pixels, title, show_roi_legend=True):
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(BG_COLOR)
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", origin="upper", interpolation="nearest")
    if roi_pixels:
        xs = [p[0] - 1 for p in roi_pixels]
        ys = [p[1] - 1 for p in roi_pixels]
        ax.scatter(xs, ys, s=7, facecolors="none", edgecolors=ROI_COLOR, linewidths=.6, zorder=3)
        if show_roi_legend:
            ax.text(.02, .02, f"o  n={len(roi_pixels)} ROI px", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.3, color=ROI_COLOR, bbox=dict(fc="white", ec="none", alpha=.75, pad=1.5))
    ax.set_title(title, fontsize=8.6, fontweight="bold", loc="left", linespacing=1.3, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def mode_footer(fig, cfg, width_chars, fontsize):
    """Mode banner as a figure footnote, wrapped to width_chars - the single-
    mode banners are long sentences that overflow a narrow figure on one line."""
    text = "\n".join(textwrap.wrap(cfg.mode_banner(), width=width_chars))
    fig.text(.5, .01, text, ha="center", va="bottom", fontsize=fontsize, color="0.4")


def roi_caption(roi):
    if roi == "per-target":
        return "Cyan outline = each target's own top-N pixels (per-target ROI, from spectra.pkl)."
    if roi == "common":
        return "Cyan outline = the shared common-ROI pixel set (from common_pixels.pkl)."
    return "No ROI overlay."


def main():
    ap = argparse.ArgumentParser(description="Plot Fig S2: per-target ion images (2D spatial intensity maps) from pass1 output.")
    add_run_dir_args(ap)
    ap.add_argument("--mode", choices=["prof", "cent"], default=None,
                     help="Which pass1 tag's peak_<tag>.npy/coords_<tag>.pkl to render (default: prof if Profile Mode is available, else cent)")
    ap.add_argument("--roi", choices=["per-target", "common", "none"], default="per-target",
                     help="ROI overlay: each target's own top-N pixels (per-target, default), the shared common ROI, or none")
    ap.add_argument("--vmax-percentile", type=float, default=99.0,
                     help="Percentile (of non-NaN, above-floor pixels) used as the colour scale max (default 99)")
    ap.add_argument("--vmin", type=float, default=0.0, help="Colour scale minimum (default 0)")
    ap.add_argument("--no-roi-legend", action="store_true", help="Hide the ROI pixel-count caption on each panel")
    args = ap.parse_args()

    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets
    show_roi_legend = not args.no_roi_legend

    mode = args.mode or ("prof" if cfg.has_profile else "cent")
    which_mode_folder = "Profile Mode" if mode == "prof" else "Centroid Mode"
    peak_path = Path(cfg.out(f"peak_{mode}.npy"))
    coords_path = Path(cfg.out(f"coords_{mode}.pkl"))
    if not peak_path.is_file() or not coords_path.is_file():
        raise RunConfigError(
            f"--mode {mode} needs {peak_path.name} and {coords_path.name} in {cfg.out_dir}, "
            f"but they don't exist yet - run pass1.py for {which_mode_folder} first"
        )
    peak = np.load(peak_path)
    with open(coords_path, "rb") as f:
        coords = pickle.load(f)

    panels = []
    for ti, T in enumerate(TARGETS):
        grid = build_grid(coords, peak[ti])
        vmax = compute_vmax(grid, cfg.intensity_floor, args.vmax_percentile)
        if vmax <= args.vmin:
            vmax = args.vmin + 1e-9
        roi_pixels = load_roi_pixels(cfg, args.roi, T)
        panels.append((T, grid, vmax, roi_pixels))

    cbar_label = f"Intensity (a.u.)  ·  colour capped at {args.vmax_percentile:g}th percentile"

    # individual PNGs
    for T, grid, vmax, roi_pixels in panels:
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        im = render_panel(ax, grid, args.vmin, vmax, roi_pixels, target_title(cfg, T), show_roi_legend=show_roi_legend)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label(cbar_label, fontsize=7)
        mode_footer(fig, cfg, width_chars=95, fontsize=5.5)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(cfg.out(f"ion_image_mz{T:.4f}.png"), dpi=300)
        plt.close(fig)

    # combined multi-panel figure, shared layout conventions with Fig_S1
    ncols = min(3, len(TARGETS))
    nrows = math.ceil(len(TARGETS) / ncols)
    fig = plt.figure(figsize=(4.4 * ncols, 1.1 + 3.9 * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[.34] + [1] * nrows,
                          hspace=.5, wspace=.35, left=.05, right=.97, top=.93, bottom=.06)
    hd = fig.add_subplot(gs[0, :]); hd.axis("off")
    hd.text(0, .72, "Ion images: per-pixel intensity within the target detection window", fontsize=13, fontweight="bold", va="top")
    hd.text(0, .30, f"{cfg.sample_name}  ·  {cfg.instrument_desc}\n"
                     f"Colour = per-pixel intensity (a.u.), capped at the {args.vmax_percentile:g}th percentile of above-floor pixels\n"
                     f"(not the maximum) so a single hot pixel doesn't flatten the scale.   Grey = pixel not sampled.\n"
                     + roi_caption(args.roi),
            fontsize=7.6, va="top", color="0.3", linespacing=1.7)
    for i, (T, grid, vmax, roi_pixels) in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r + 1, c])
        im = render_panel(ax, grid, args.vmin, vmax, roi_pixels, target_title(cfg, T), show_roi_legend=show_roi_legend)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label(cbar_label, fontsize=6.3)
    mode_footer(fig, cfg, width_chars=32 * ncols, fontsize=6)
    fig.savefig(cfg.out("Fig_S2_ion_images.png"), dpi=300)
    fig.savefig(cfg.out("Fig_S2_ion_images.pdf"))
    plt.close(fig)
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except RunConfigError as e:
        raise SystemExit(str(e))
