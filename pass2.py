import warnings; warnings.filterwarnings("ignore")
import argparse
from contextlib import ExitStack
import numpy as np, pickle
from msi_io import RunConfig, RunConfigError, add_run_dir_args, read_binary_array, open_parser_and_ibd


def main():
    ap = argparse.ArgumentParser(description="Pass 2: average profile+centroid spectra over each target's own top-N pixels.")
    add_run_dir_args(ap)
    args = ap.parse_args()

    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets
    NTOP, HALFWIN, GRID = cfg.ntop, cfg.halfwin, cfg.grid
    have_P, have_C = cfg.has_profile, cfg.has_centroid

    with ExitStack() as stack:
        P = C = bP = bC = None
        if have_P:
            P, bP = stack.enter_context(open_parser_and_ibd(cfg.imzml_path("Profile Mode")))
        if have_C:
            C, bC = stack.enter_context(open_parser_and_ibd(cfg.imzml_path("Centroid Mode")))
        cP = [(c[0], c[1]) for c in P.coordinates] if have_P else []
        cC = [(c[0], c[1]) for c in C.coordinates] if have_C else []
        if have_P and have_C:
            print("coord overlap:", len(set(cP) & set(cC)), "prof only:", len(set(cP) - set(cC)))

        # Rank pixels using whichever mode has per-pixel intensities from pass1;
        # prefer profile when both exist, so both-mode output is unchanged.
        rank_is_profile = have_P
        rank_coords = cP if rank_is_profile else cC
        peakR = np.load(cfg.out("peak_prof.npy" if rank_is_profile else "peak_cent.npy"))
        # rank-mode pixel coordinate -> the OTHER mode's own pixel index (only
        # meaningful when both modes are present; the rank mode is indexed directly)
        cmap = {}
        if have_P and have_C:
            other_coords = cC if rank_is_profile else cP
            cmap = {xy: i for i, xy in enumerate(other_coords)}

        def getspec(p, buf, k):
            mo, ml = p.mzOffsets[k], p.mzLengths[k]; io, il = p.intensityOffsets[k], p.intensityLengths[k]
            mz = np.array(read_binary_array(buf, mo, ml, p.mzPrecision), dtype=np.float64)
            it = np.array(read_binary_array(buf, io, il, p.intensityPrecision), dtype=np.float64)
            return mz, it

        out = {}
        for ti, T in enumerate(TARGETS):
            order = np.argsort(peakR[ti])[::-1][:NTOP]
            order = [k for k in order if peakR[ti, k] > cfg.intensity_floor]
            grid = np.arange(T - HALFWIN, T + HALFWIN, GRID)
            acc = np.zeros_like(grid); nprof = 0
            sticks_mz = []; sticks_it = []; ncent = 0
            for k in order:
                if have_P:
                    pk = k if rank_is_profile else cmap.get(rank_coords[k])
                    if pk is not None:
                        mz, it = getspec(P, bP, pk)
                        a = np.searchsorted(mz, T - HALFWIN); b = np.searchsorted(mz, T + HALFWIN)
                        if b - a > 2:
                            acc += np.interp(grid, mz[a:b], it[a:b], left=0, right=0); nprof += 1
                if have_C:
                    ck = k if not rank_is_profile else cmap.get(rank_coords[k])
                    if ck is not None:
                        mzc, itc = getspec(C, bC, ck)
                        a2 = np.searchsorted(mzc, T - HALFWIN); b2 = np.searchsorted(mzc, T + HALFWIN)
                        if b2 > a2:
                            sticks_mz.append(mzc[a2:b2]); sticks_it.append(itc[a2:b2]); ncent += 1
            prof = acc / max(nprof, 1)
            smz = np.concatenate(sticks_mz) if sticks_mz else np.array([])
            sit = np.concatenate(sticks_it) if sticks_it else np.array([])
            # cluster sticks at 4 ppm
            o = np.argsort(smz); smz = smz[o]; sit = sit[o]
            cl_mz = []; cl_it = []; cl_n = []
            i = 0
            while i < len(smz):
                j = i
                while j + 1 < len(smz) and (smz[j + 1] - smz[i]) / smz[i] * 1e6 < 4.0: j += 1
                w = sit[i:j + 1]
                cl_mz.append(np.average(smz[i:j + 1], weights=w) if w.sum() > 0 else smz[i:j + 1].mean())
                cl_it.append(w.sum() / max(ncent, 1)); cl_n.append(j - i + 1)
                i = j + 1
            out[T] = dict(grid=grid, prof=prof, cmz=np.array(cl_mz), cit=np.array(cl_it),
                          cn=np.array(cl_n), nprof=nprof, ncent=ncent, modes=cfg.modes,
                          pixels=[rank_coords[k] for k in order], peaks=[float(peakR[ti, k]) for k in order])
            print(T, "nprof", nprof, "ncent", ncent, "clusters", len(cl_mz), "profmax", prof.max())
    pickle.dump(out, open(cfg.out("spectra.pkl"), "wb"))
    print("SAVED")


if __name__ == "__main__":
    try:
        main()
    except RunConfigError as e:
        raise SystemExit(str(e))
