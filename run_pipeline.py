"""Chain pass1 -> common -> pass2 -> metrics -> plot across one run folder, or every
run folder found under a parent directory of data runs.

A "run folder" is any directory containing a usable 'Profile Mode' and/or
'Centroid Mode' subfolder (each holding one .imzML + .ibd pair) - either one
alone is enough to process a run, with reduced metrics; see RunConfig.mode_banner().
It may also directly hold a single .imzML + .ibd pair with no wrapper
subfolder, in which case the mode is auto-detected from the file itself.
Point this at either:
  - a single run folder, or
  - a parent folder holding many run folders (each processed in turn)

Examples:
  python run_pipeline.py "/data/Neg_20240306"
  python run_pipeline.py "/data/all_runs" --keep-going
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from msi_io import MODES, RunConfig, RunConfigError, mode_status

HERE = Path(__file__).resolve().parent


def has_modes(p):
    """True if at least one acquisition-mode subfolder of p is usable, or p
    itself directly holds a usable .imzML + .ibd pair (flat layout, mode
    auto-detected from content - see RunConfig._detect_modes). Does not raise
    on a present-but-broken mode folder - that only matters once a run is
    actually selected for processing, where RunConfig raises loudly for it."""
    if any(mode_status(p / m)[0] == "ok" for m in MODES):
        return True
    return mode_status(p)[0] == "ok"


def resolve_out_dir(run_dir, out_dir):
    return Path(out_dir) if out_dir else Path(run_dir) / "output"


def clean_out_dir(run_dir, out_dir):
    """Wipe any previous output before a fresh run. Every artifact here is a
    full regeneration from the current target list."""
    d = resolve_out_dir(run_dir, out_dir)
    if d.is_dir():
        shutil.rmtree(d)


def discover_runs(path):
    if has_modes(path):
        return [path]
    candidates = sorted(c for c in path.iterdir() if c.is_dir() and has_modes(c))
    if not candidates:
        raise SystemExit(f"no run folders (with a usable 'Profile Mode' and/or 'Centroid Mode' subfolder) found at or under {path}")
    return candidates


def stage_commands(run_dir, out_dir, targets_path, show_metrics_box=True, cfg=None):
    """The ordered (label, cmd) pairs that make up one full run. Shared by the
    CLI driver below and the Streamlit GUI (app.py), so both stay in sync.
    Pass1 only runs for modes actually present; common/pass2/metrics/plot
    always run, operating on whatever pass1 output exists. Raises
    RunConfigError (via RunConfig) if run_dir isn't a usable/valid run folder."""
    if cfg is None:
        cfg = RunConfig(run_dir, out_dir, targets_path)
    tail = []
    if out_dir: tail += ["--out", str(out_dir)]
    if targets_path: tail += ["--targets", str(targets_path)]
    box_flag = [] if show_metrics_box else ["--no-metrics-box"]

    cmds = []
    if cfg.has_profile:
        cmds.append(("pass1 (Profile Mode)", [sys.executable, str(HERE / "pass1.py"), str(run_dir), "Profile Mode", "prof"] + tail))
    if cfg.has_centroid:
        cmds.append(("pass1 (Centroid Mode)", [sys.executable, str(HERE / "pass1.py"), str(run_dir), "Centroid Mode", "cent"] + tail))
    cmds += [
        ("common", [sys.executable, str(HERE / "common.py"), str(run_dir)] + tail),
        ("pass2", [sys.executable, str(HERE / "pass2.py"), str(run_dir)] + tail),
        ("metrics", [sys.executable, str(HERE / "metrics.py"), str(run_dir)] + tail),
        ("metrics (common)", [sys.executable, str(HERE / "metrics.py"), str(run_dir), "--common"] + tail),
        ("plot", [sys.executable, str(HERE / "plot.py"), str(run_dir)] + tail + box_flag),
        ("plot (common)", [sys.executable, str(HERE / "plot_c.py"), str(run_dir)] + tail + box_flag),
        ("plot (common, clean)", [sys.executable, str(HERE / "plot_cc.py"), str(run_dir)] + tail),
    ]
    return cmds


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def process_run(run_dir, out_dir, targets_path, show_metrics_box=True):
    clean_out_dir(run_dir, out_dir)
    cfg = RunConfig(run_dir, out_dir, targets_path)
    print(cfg.mode_banner(), flush=True)
    for _label, cmd in stage_commands(run_dir, out_dir, targets_path, show_metrics_box, cfg=cfg):
        run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="A single run folder, or a parent folder containing multiple run folders")
    ap.add_argument("--out", default=None, help="Output folder for a single run (default: <run_dir>/output). Ignored when processing multiple runs (each uses <run_dir>/output).")
    ap.add_argument("--targets", default=None, help="targets.json to use for every run processed (default: each run's own <run_dir>/targets.json, else built-in defaults)")
    ap.add_argument("--keep-going", action="store_true", help="If one run folder fails, continue with the rest instead of stopping")
    ap.add_argument("--no-metrics-box", action="store_true", help="Hide the FWHM/R/peak-count text box on the generated figures")
    args = ap.parse_args()

    path = Path(args.path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"not a folder: {path}")
    runs = discover_runs(path)
    print(f"found {len(runs)} run folder(s):")
    for r in runs: print(f"  {r}")

    out_dir = args.out if len(runs) == 1 else None
    failed = []
    for r in runs:
        try:
            process_run(r, out_dir, args.targets, show_metrics_box=not args.no_metrics_box)
        except (subprocess.CalledProcessError, RunConfigError) as e:
            print(f"!!! run failed: {r} ({e})", file=sys.stderr)
            failed.append(r)
            if not args.keep_going:
                raise SystemExit(1)

    if failed:
        print(f"\n{len(failed)} of {len(runs)} run(s) failed:")
        for r in failed: print(f"  {r}")
        raise SystemExit(1)
    print(f"\nall {len(runs)} run(s) completed.")


if __name__ == "__main__":
    main()
