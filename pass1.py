import warnings; warnings.filterwarnings("ignore")
import argparse, time
import numpy as np, pickle
from msi_io import RunConfig, RunConfigError, add_run_dir_args, read_binary_array, open_parser_and_ibd


def main():
    ap = argparse.ArgumentParser(description="Pass 1: per-pixel max intensity within +/-PPM of each target m/z, for one acquisition mode.")
    add_run_dir_args(ap)
    ap.add_argument("mode", choices=["Profile Mode", "Centroid Mode"], help="Which subfolder/acquisition mode to scan")
    ap.add_argument("tag", help="Tag used to name output files, e.g. 'prof' or 'cent'")
    args = ap.parse_args()

    cfg = RunConfig(args.run_dir, args.out, args.targets)
    TARGETS = cfg.targets
    PPM = cfg.ppm

    with open_parser_and_ibd(cfg.imzml_path(args.mode)) as (p, buf):
        n = len(p.coordinates); print(args.tag, n, flush=True)
        peak = np.zeros((len(TARGETS), n), dtype=np.float32)
        mzo = np.array(p.mzOffsets); mzl = np.array(p.mzLengths)
        io = np.array(p.intensityOffsets)
        it_itemsize = np.dtype(p.intensityPrecision).itemsize
        t0 = time.time()
        for k in range(n):
            L = mzl[k]
            if L == 0: continue
            o = mzo[k]
            mz = read_binary_array(buf, o, L, p.mzPrecision)
            for ti, T in enumerate(TARGETS):
                tol = T * PPM * 1e-6
                a = np.searchsorted(mz, T - tol); b = np.searchsorted(mz, T + tol)
                if b > a:
                    peak[ti, k] = read_binary_array(buf, io[k] + a * it_itemsize, b - a, p.intensityPrecision).max()
            if k % 4000 == 0: print(k, round(time.time() - t0, 1), flush=True)
        print(args.tag, "DONE", round(time.time() - t0, 1), flush=True)
        coords = [(c[0], c[1]) for c in p.coordinates]

    np.save(cfg.out(f"peak_{args.tag}.npy"), peak)
    with open(cfg.out(f"coords_{args.tag}.pkl"), "wb") as f:
        pickle.dump(coords, f)
    for ti, T in enumerate(TARGETS):
        above_floor = int((peak[ti] > cfg.intensity_floor).sum())
        print(f"  {T}: {above_floor} px above intensity floor ({cfg.intensity_floor:g}), max {peak[ti].max():.0f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RunConfigError as e:
        raise SystemExit(str(e))
