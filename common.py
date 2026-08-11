import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pickle
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import rankdata
from msi_io import RunConfig, add_run_dir_args


def main():
    ap = argparse.ArgumentParser(description="Pick one common set of top-N pixels good across all targets, and average profile+centroid spectra over it.")
    add_run_dir_args(ap)
    args = ap.parse_args()

    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets
    NTOP, HALFWIN, GRID = cfg.ntop, cfg.halfwin, cfg.grid

    P = ImzMLParser(str(cfg.imzml_path("Profile Mode")))
    C = ImzMLParser(str(cfg.imzml_path("Centroid Mode")))
    bP = np.memmap(P.filename.replace(".imzML", ".ibd"), dtype=np.uint8, mode='r')
    bC = np.memmap(C.filename.replace(".imzML", ".ibd"), dtype=np.uint8, mode='r')
    cP = [(c[0], c[1]) for c in P.coordinates]; cmap = {(c[0], c[1]): i for i, c in enumerate(C.coordinates)}
    pk = np.load(cfg.out("peak_prof.npy"))

    # pixels must have signal for all targets
    ok = (pk > 0).all(axis=0)
    print("pixels with all", len(TARGETS), "detected:", ok.sum(), "of", pk.shape[1])
    # percentile rank within the qualifying pixels, per metabolite
    R = np.zeros((len(TARGETS), pk.shape[1]))
    for i in range(len(TARGETS)):
        r = np.full(pk.shape[1], np.nan); r[ok] = rankdata(pk[i, ok]) / ok.sum()
        R[i] = r
    score = np.where(ok, R.mean(axis=0), -1)          # balanced across all targets
    worst = np.where(ok, np.nanmin(R, axis=0), -1)     # conservative
    sel = np.argsort(score)[::-1][:NTOP]
    print("common ROI: mean pct-rank %.3f  worst-metabolite pct-rank >= %.3f" % (score[sel].mean(), worst[sel].min()))
    for i, T in enumerate(TARGETS):
        old = np.argsort(pk[i])[::-1][:NTOP]
        print(f"  {T}: overlap with its own top-{NTOP} = {len(set(sel) & set(old))}   "
              f"mean int {pk[i, sel].mean():10.1f} vs {pk[i, old].mean():10.1f} ({pk[i, sel].mean() / pk[i, old].mean() * 100:.0f}%)")

    def getspec(p, buf, k):
        mo, ml = p.mzOffsets[k], p.mzLengths[k]; io, il = p.intensityOffsets[k], p.intensityLengths[k]
        return (np.array(buf[mo:mo + ml * 4].view(np.float32), dtype=np.float64),
                np.array(buf[io:io + il * 4].view(np.float32), dtype=np.float64))

    out = {}
    for ti, T in enumerate(TARGETS):
        grid = np.arange(T - HALFWIN, T + HALFWIN, GRID); acc = np.zeros_like(grid); n1 = 0
        smzs = []; sits = []; n2 = 0
        for k in sel:
            mz, it = getspec(P, bP, k)
            a = np.searchsorted(mz, T - HALFWIN); b = np.searchsorted(mz, T + HALFWIN)
            if b - a > 2: acc += np.interp(grid, mz[a:b], it[a:b], left=0, right=0); n1 += 1
            j = cmap.get(cP[k])
            if j is not None:
                mzc, itc = getspec(C, bC, j)
                a2 = np.searchsorted(mzc, T - HALFWIN); b2 = np.searchsorted(mzc, T + HALFWIN)
                if b2 > a2: smzs.append(mzc[a2:b2]); sits.append(itc[a2:b2]); n2 += 1
        prof = acc / max(n1, 1)
        smz = np.concatenate(smzs) if smzs else np.array([]); sit = np.concatenate(sits) if sits else np.array([])
        o = np.argsort(smz); smz = smz[o]; sit = sit[o]
        cm = []; ci = []; cn = []; i = 0
        while i < len(smz):
            j = i
            while j + 1 < len(smz) and (smz[j + 1] - smz[i]) / smz[i] * 1e6 < 4.0: j += 1
            w = sit[i:j + 1]
            cm.append(np.average(smz[i:j + 1], weights=w) if w.sum() > 0 else smz[i:j + 1].mean())
            ci.append(w.sum() / max(n2, 1)); cn.append(j - i + 1); i = j + 1
        out[T] = dict(grid=grid, prof=prof, cmz=np.array(cm), cit=np.array(ci), cn=np.array(cn),
                      nprof=n1, ncent=n2, pixels=[cP[k] for k in sel])
    pickle.dump(out, open(cfg.out("spectra_common.pkl"), "wb"))
    pickle.dump([cP[k] for k in sel], open(cfg.out("common_pixels.pkl"), "wb"))
    print("SAVED")


if __name__ == "__main__":
    main()
