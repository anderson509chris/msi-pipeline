import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pickle
from pyimzml.ImzMLParser import ImzMLParser
from msi_io import RunConfig, add_run_dir_args, read_binary_array


def main():
    ap = argparse.ArgumentParser(description="Pass 2: average profile+centroid spectra over each target's own top-N pixels.")
    add_run_dir_args(ap)
    args = ap.parse_args()

    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets
    NTOP, HALFWIN, GRID = cfg.ntop, cfg.halfwin, cfg.grid

    P = ImzMLParser(str(cfg.imzml_path("Profile Mode")))
    C = ImzMLParser(str(cfg.imzml_path("Centroid Mode")))
    bP = np.memmap(P.filename.replace(".imzML", ".ibd"), dtype=np.uint8, mode='r')
    bC = np.memmap(C.filename.replace(".imzML", ".ibd"), dtype=np.uint8, mode='r')
    cP = [(c[0], c[1]) for c in P.coordinates]; cC = [(c[0], c[1]) for c in C.coordinates]
    cmap = {xy: i for i, xy in enumerate(cC)}
    peakP = np.load(cfg.out("peak_prof.npy"))
    print("coord overlap:", len(set(cP) & set(cC)), "prof only:", len(set(cP) - set(cC)))

    def getspec(p, buf, k):
        mo, ml = p.mzOffsets[k], p.mzLengths[k]; io, il = p.intensityOffsets[k], p.intensityLengths[k]
        mz = np.array(read_binary_array(buf, mo, ml, p.mzPrecision), dtype=np.float64)
        it = np.array(read_binary_array(buf, io, il, p.intensityPrecision), dtype=np.float64)
        return mz, it

    out = {}
    for ti, T in enumerate(TARGETS):
        order = np.argsort(peakP[ti])[::-1][:NTOP]
        order = [k for k in order if peakP[ti, k] > cfg.intensity_floor]
        grid = np.arange(T - HALFWIN, T + HALFWIN, GRID)
        acc = np.zeros_like(grid); nprof = 0
        sticks_mz = []; sticks_it = []; ncent = 0
        for k in order:
            mz, it = getspec(P, bP, k)
            a = np.searchsorted(mz, T - HALFWIN); b = np.searchsorted(mz, T + HALFWIN)
            if b - a > 2:
                acc += np.interp(grid, mz[a:b], it[a:b], left=0, right=0); nprof += 1
            j = cmap.get(cP[k])
            if j is not None:
                mzc, itc = getspec(C, bC, j)
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
                      cn=np.array(cl_n), nprof=nprof, ncent=ncent,
                      pixels=[cP[k] for k in order], peaks=[float(peakP[ti, k]) for k in order])
        print(T, "nprof", nprof, "ncent", ncent, "clusters", len(cl_mz), "profmax", prof.max())
    pickle.dump(out, open(cfg.out("spectra.pkl"), "wb"))
    print("SAVED")


if __name__ == "__main__":
    main()
