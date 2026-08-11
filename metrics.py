import argparse
import pickle, numpy as np
from msi_io import RunConfig, add_run_dir_args


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
        tol = T * PPM * 1e-6
        warn = []

        # profile apex nearest target
        m = (g >= T - 6 * tol) & (g <= T + 6 * tol)
        ia = np.argmax(np.where(m, y, -1)); apex = g[ia]; ymax = y[ia]

        if ymax <= 0:
            res[T] = dict(name=NAMES[T], apex=apex, ymax=0.0, fwhm=np.nan, R=np.nan, pcen=np.nan,
                           ppm_prof=np.nan, cmz=np.nan, cit=np.nan, ppm_cent=np.nan,
                           areafrac=np.nan, nb_ppm=np.nan, nb_mz=np.nan, nb_it=np.nan,
                           n_in_window=0, inwin=[], ndet=0, nprof=s["nprof"],
                           warning="no signal above intensity floor")
            print(f"{T}: NO SIGNAL above intensity floor - metrics set to NaN")
            continue

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

        # centroid stick nearest target
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
            warn.append("no centroid peak detected near target")
            cmz = cit = np.nan; nb = (np.nan, np.nan, np.nan); inwin = []; ndet = 0

        res[T] = dict(name=NAMES[T], apex=apex, ymax=ymax, fwhm=fwhm, R=T / fwhm, pcen=pcen,
                       ppm_prof=(pcen - T) / T * 1e6, cmz=cmz, cit=cit, ppm_cent=(cmz - T) / T * 1e6,
                       areafrac=area_in / area_pk, nb_ppm=nb[0], nb_mz=nb[1], nb_it=nb[2],
                       n_in_window=len(inwin), inwin=inwin, ndet=ndet, nprof=s["nprof"],
                       warning="; ".join(warn))
        nb_rel = nb[2] / cit if not np.isnan(cit) else np.nan
        print(f"{T}: apex {apex:.5f} R {T / fwhm:,.0f} FWHM {fwhm * 1000:.2f} mDa | cent {cmz:.5f} ({(cmz - T) / T * 1e6:+.2f} ppm) | areafrac {area_in / area_pk:.3f} | nearest other stick {nb[0]:.1f} ppm @ {nb[1]:.5f} rel {nb_rel:.4f} | sticks in +/-{PPM:g}ppm: {len(inwin)}"
              + (f"  [WARNING: {res[T]['warning']}]" if warn else ""))
    pickle.dump(res, open(cfg.out(metrics_name), "wb"))


if __name__ == "__main__":
    main()
