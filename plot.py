import warnings; warnings.filterwarnings("ignore")
import argparse
import pickle, numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from msi_io import RunConfig, add_run_dir_args

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False})
CP = "#1f4e79"; CC = "#c0392b"; CB = "#f2b705"


def panel(axL, axR, T, letter, d, mt):
    s = d[T]; m = mt[T]; g = s["grid"]; y = s["prof"]
    tol = T * 3e-6
    ymax = m["ymax"]
    for ax, halfppm in ((axL, None), (axR, 25)):
        if halfppm is None:
            lo, hi = T - 0.06, T + 0.06
        else:
            lo, hi = T - T * halfppm * 1e-6, T + T * halfppm * 1e-6
        w = (g >= lo) & (g <= hi)
        ax.axvspan(T - tol, T + tol, color=CB, alpha=.28, lw=0, zorder=0)
        ax.plot(g[w], y[w], color=CP, lw=1.1, zorder=3)
        ax.fill_between(g[w], 0, y[w], color=CP, alpha=.10, zorder=2)
        sel = (s["cmz"] >= lo) & (s["cmz"] <= hi) & (s["cit"] > 0)
        ax.vlines(s["cmz"][sel], 0, s["cit"][sel], color=CC, lw=1.4, zorder=4)
        ax.axvline(T, color="0.35", lw=.7, ls=(0, (4, 3)), zorder=1)
        if halfppm is None:
            ax.plot(g[w], np.minimum(y[w] * 100, ymax * 1.05), color="0.55", lw=.7, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(0, ymax * (1.30 if halfppm is None else 1.42))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_size(6.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5 if halfppm is None else 4))
    axL.set_ylabel("Mean ion intensity\n(counts)", fontsize=7.5)
    axL.set_xlabel("m/z", fontsize=8); axR.set_xlabel("m/z", fontsize=8)
    axR.tick_params(labelsize=7)
    axL.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.2f}"))
    axR.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.4f}"))
    axL.set_title(f"{letter}   m/z {T:.4f}", loc="left", fontsize=9.2, fontweight="bold", pad=4)
    axR.set_title("zoom ±25 ppm", loc="left", fontsize=7.5, color="0.35", pad=4)
    axL.text(.985, .96, f"±0.06 Da window · n = {s['nprof']} pixels\ngrey = same trace ×100 (baseline check)",
             transform=axL.transAxes, ha="right", va="top", fontsize=6.6, color="0.35", linespacing=1.5)
    txt = (f"centroid  {m['cmz']:.4f}  ({m['ppm_cent']:+.2f} ppm)\n"
           f"FWHM  {m['fwhm'] * 1000:.2f} mDa   R = {m['R']:,.0f}\n"
           f"peaks in ±3 ppm: {m['n_in_window']}   nearest other peak: {m['nb_ppm']:,.0f} ppm")
    axR.text(.02, .96, txt, transform=axR.transAxes, ha="left", va="top", fontsize=6.6,
             color="0.15", linespacing=1.5,
             bbox=dict(fc="white", ec="0.8", lw=.5, boxstyle="round,pad=0.35", alpha=.92))


def main():
    ap = argparse.ArgumentParser(description="Plot Fig S1: raw profile waveform vs centroid sticks, over each target's own top-N pixels.")
    add_run_dir_args(ap)
    args = ap.parse_args()
    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets

    d = pickle.load(open(cfg.out("spectra.pkl"), "rb")); mt = pickle.load(open(cfg.out("metrics.pkl"), "rb"))

    fig = plt.figure(figsize=(10.6, 12.4))
    gs = fig.add_gridspec(len(TARGETS) + 1, 2, height_ratios=[.30] + [1] * len(TARGETS), width_ratios=[1.35, 1],
                          hspace=.52, wspace=.20, left=.075, right=.975, top=.965, bottom=.048)
    hd = fig.add_subplot(gs[0, :]); hd.axis("off")
    hd.text(0, .72, "Local mass spectra: raw profile waveform vs. METASPACE-style centroid sticks",
            fontsize=13, fontweight="bold", va="top")
    hd.text(0, .34, f"{cfg.sample_name}  ·  {cfg.instrument_desc}\n"
                     "Blue = mean raw profile spectrum over the top-N highest-intensity pixels.   Red sticks = mean centroided peak list (non-zero) over the same pixels.\n"
                     "Gold band = ±3 ppm integration window.   Dashed line = theoretical m/z.",
            fontsize=7.6, va="top", color="0.3", linespacing=1.7)
    for i, (T, L) in enumerate(zip(TARGETS, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
        panel(fig.add_subplot(gs[i + 1, 0]), fig.add_subplot(gs[i + 1, 1]), T, L, d, mt)
    fig.savefig(cfg.out("Fig_S1_profile_vs_centroid.pdf"))
    fig.savefig(cfg.out("Fig_S1_profile_vs_centroid.png"), dpi=300)
    plt.close(fig)

    # individual PNGs
    for T, L in zip(TARGETS, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        f = plt.figure(figsize=(9.4, 3.3))
        g2 = f.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=.22, left=.085, right=.98, top=.85, bottom=.17)
        panel(f.add_subplot(g2[0, 0]), f.add_subplot(g2[0, 1]), T, L, d, mt)
        f.savefig(cfg.out(f"spectrum_mz{T:.4f}.png"), dpi=300); plt.close(f)

    with open(cfg.out("peak_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["assignment", "formula_neutral", "ion", "theoretical_mz", "profile_apex_mz",
                    "profile_centroid_mz", "profile_mass_error_ppm", "centroid_mode_mz", "centroid_mass_error_ppm",
                    "FWHM_mDa", "resolving_power", "frac_peak_area_within_3ppm", "n_centroid_peaks_within_3ppm",
                    "nearest_other_real_peak_ppm", "nearest_other_rel_intensity", "n_pixels_averaged",
                    "pixels_with_signal"])
        pk = np.load(cfg.out("peak_prof.npy"))
        for i, T in enumerate(TARGETS):
            m = mt[T]; nm = cfg.names[T]; _, fp = cfg.formulas[T]
            w.writerow([nm, fp, "[M-H]-", f"{T:.4f}", f"{m['apex']:.5f}", f"{m['pcen']:.5f}",
                        f"{m['ppm_prof']:+.2f}", f"{m['cmz']:.5f}", f"{m['ppm_cent']:+.2f}",
                        f"{m['fwhm'] * 1000:.2f}", f"{m['R']:.0f}", f"{m['areafrac']:.3f}", m['n_in_window'],
                        f"{m['nb_ppm']:.1f}", f"{m['nb_it'] / m['cit']:.5f}", m['nprof'], int((pk[i] > 0).sum())])
    print("OK")


if __name__ == "__main__":
    main()
