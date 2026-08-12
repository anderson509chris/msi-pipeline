import argparse
import pickle, numpy as np
from msi_io import RunConfig, RunConfigError, add_run_dir_args


def main():
    ap = argparse.ArgumentParser(description="Compute per-target peak-shape metrics (apex, FWHM, centroid mass error, etc).")
    add_run_dir_args(ap)
    ap.add_argument("--common", action="store_true", help="Read spectra_common.pkl / write metrics_common.pkl instead of spectra.pkl / metrics.pkl")
    ap.add_argument("--spectra", default=None, help="Override input spectra pkl filename")
    ap.add_argument("--metrics-out", default=None, help="Override output metrics pkl filename")
    args = ap.parse_args()

    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets
    NAMES = cfg.names
    PPM = cfg.ppm

    spectra_name = args.spectra or ("spectra_common.pkl" if args.common else "spectra.pkl")
    metrics_name = args.metrics_out or ("metrics_common.pkl" if args.common else "metrics.pkl")

    d = pickle.load(open(cfg.out(spectra_name), "rb"))
    res = {}
    for T in TARGETS:
        s = d[T]; g = s["grid"]; y = s["prof"]
        modes = s.get("modes", cfg.modes)
        tol = T * PPM * 1e-6
        warn = []

        # --- profile-side: apex, FWHM, resolving power, profile centroid, area fraction ---
        # Two distinct causes give an empty profile trace, and they mean very
        # different things to a user: the mode folder genuinely isn't part of
        # this run (nothing to investigate) vs. the mode is present but this
        # particular target has no signal above the intensity floor (a real,
        # possibly-interesting absence). Report them separately rather than
        # conflating both into "profile mode not available", which sends
        # someone with perfectly good profile data looking for a folder
        # problem that doesn't exist.
        has_profile_mode = "Profile Mode" in modes
        has_profile_data = has_profile_mode and s.get("nprof", 0) > 0 and y.max() > 0
        if not has_profile_data:
            warn.append("profile mode not available" if not has_profile_mode else "no profile signal above intensity floor")
            apex = np.nan; ymax = np.nan; fwhm = np.nan; R = np.nan; pcen = np.nan
            ppm_prof = np.nan; areafrac = np.nan
        else:
            # profile apex nearest target
            m = (g >= T - 6 * tol) & (g <= T + 6 * tol)
            ia = np.argmax(np.where(m, y, -1)); apex = g[ia]; ymax = y[ia]

            # FWHM by interpolation around apex; NaN if the half-max point isn't
            # reached inside the window (peak sits on a shoulder / window too narrow)
            half = ymax / 2
            li = ia
            while li > 0 and y[li] > half: li -= 1
            ri = ia
            while ri < len(y) - 1 and y[ri] > half: ri += 1
            half_max_reached = not (li == 0 and y[li] > half) and not (ri == len(y) - 1 and y[ri] > half)

            def xint(i1, i2):
                if y[i2] == y[i1]: return g[i1]
                return g[i1] + (half - y[i1]) * (g[i2] - g[i1]) / (y[i2] - y[i1])

            if half_max_reached:
                lo = xint(li, li + 1); hi = xint(ri, ri - 1); fwhm = hi - lo
                # profile centroid within FWHM-ish region
                mm = (g >= apex - 2 * fwhm) & (g <= apex + 2 * fwhm)
                pcen = np.average(g[mm], weights=y[mm]) if y[mm].sum() > 0 else np.nan
                area_pk = np.trapezoid(y[mm], g[mm])
            else:
                warn.append("half-max not reached within window; FWHM/R/profile centroid undetermined")
                fwhm = np.nan; pcen = np.nan; area_pk = np.nan

            # area fraction inside +/-PPM of target
            win = (g >= T - tol) & (g <= T + tol)
            area_in = np.trapezoid(y[win], g[win])
            R = T / fwhm
            ppm_prof = (pcen - T) / T * 1e6
            areafrac = area_in / area_pk if (not np.isnan(area_pk) and area_pk != 0) else np.nan

        # --- centroid-side: nearest stick, neighbour, in-window count ---
        # mirrors the profile-side above: distinguish an absent Centroid Mode
        # folder from one that's present but simply didn't detect a peak here.
        has_centroid_mode = "Centroid Mode" in modes
        nzm = s["cit"] > 0
        CMZ = s["cmz"][nzm]; CIT = s["cit"][nzm]; CN = s["cn"][nzm]
        s = dict(s); s["cmz"] = CMZ; s["cit"] = CIT; s["cn"] = CN
        if len(CMZ) > 0:
            ci = np.argmin(np.abs(CMZ - T)); cmz = CMZ[ci]; cit = CIT[ci]
            # nearest neighbouring stick
            others = [(abs(s["cmz"][j] - cmz) / cmz * 1e6, s["cmz"][j], s["cit"][j]) for j in range(len(s["cmz"])) if j != ci]
            others.sort()
            nb = others[0] if others else (np.nan, np.nan, np.nan)
            # any other stick within +/-PPM?
            inwin = [(s["cmz"][j], s["cit"][j]) for j in range(len(s["cmz"])) if abs(s["cmz"][j] - T) <= tol]
            ndet = int(s["cn"][ci])
        else:
            warn.append("centroid mode not available" if not has_centroid_mode else "no centroid peak detected near target")
            cmz = cit = np.nan; nb = (np.nan, np.nan, np.nan); inwin = []; ndet = 0
        ppm_cent = (cmz - T) / T * 1e6

        res[T] = dict(name=NAMES[T], apex=apex, ymax=ymax, fwhm=fwhm, R=R, pcen=pcen,
                       ppm_prof=ppm_prof, cmz=cmz, cit=cit, ppm_cent=ppm_cent,
                       areafrac=areafrac, nb_ppm=nb[0], nb_mz=nb[1], nb_it=nb[2],
                       n_in_window=len(inwin), inwin=inwin, ndet=ndet, nprof=s["nprof"],
                       modes=modes, warning="; ".join(warn))
        nb_rel = nb[2] / cit if not np.isnan(cit) else np.nan
        print(f"{T}: apex {apex:.5f} R {R:,.0f} FWHM {fwhm * 1000:.2f} mDa | cent {cmz:.5f} ({ppm_cent:+.2f} ppm) | areafrac {areafrac:.3f} | nearest other stick {nb[0]:.1f} ppm @ {nb[1]:.5f} rel {nb_rel:.4f} | sticks in +/-{PPM:g}ppm: {len(inwin)}"
              + (f"  [WARNING: {res[T]['warning']}]" if warn else ""))
    pickle.dump(res, open(cfg.out(metrics_name), "wb"))


if __name__ == "__main__":
    try:
        main()
    except RunConfigError as e:
        raise SystemExit(str(e))
