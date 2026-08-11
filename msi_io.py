"""Shared config/IO helpers so each pipeline stage can be pointed at any run folder.

A "run folder" is expected to look like:

    <run_dir>/
        Profile Mode/<name>.imzML (+ .ibd)
        Centroid Mode/<name>.imzML (+ .ibd)
        targets.json        (optional, see DEFAULT_TARGETS below)

All intermediate/output artifacts (peak_*.npy, spectra*.pkl, metrics*.pkl,
figures, CSVs) are written to <run_dir>/output/ by default, or to --out if given.
"""
import json
import os
from pathlib import Path

# Fallback target list, used only if a run folder has no targets.json.
DEFAULT_TARGETS = {
    "sample_name": "EpCtrl-4-1_2_S2_SM_Neg_20240306_IT",
    "instrument_desc": "Orbitrap MALDI-MSI, negative ion mode, m/z 70–500  ·  142 × 308 px, 20 µm  ·  lock mass 157.07712",
    "targets": [
        {"mz": 140.0118, "name": "Phosphoethanolamine", "formula_tex": "C$_2$H$_8$NO$_4$P", "formula_plain": "C2H8NO4P"},
        {"mz": 146.0459, "name": "Glutamate", "formula_tex": "C$_5$H$_9$NO$_4$", "formula_plain": "C5H9NO4"},
        {"mz": 151.0261, "name": "Xanthine", "formula_tex": "C$_5$H$_4$N$_4$O$_2$", "formula_plain": "C5H4N4O2"},
        {"mz": 215.0328, "name": "Glucose", "formula_tex": "C$_5$H$_{13}$O$_7$P", "formula_plain": "C5H13O7P"},
    ],
    "params": {"ntop": 100, "halfwin": 0.06, "grid": 1e-5, "ppm": 3.0},
}


class RunConfig:
    def __init__(self, run_dir, out_dir=None, targets_path=None):
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise SystemExit(f"run folder not found: {self.run_dir}")
        self.out_dir = Path(out_dir).expanduser().resolve() if out_dir else self.run_dir / "output"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = Path(targets_path).expanduser().resolve() if targets_path else self.run_dir / "targets.json"
        if cfg_path.is_file():
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = DEFAULT_TARGETS

        self.sample_name = cfg.get("sample_name", self.run_dir.name)
        self.instrument_desc = cfg.get("instrument_desc", "")
        self.target_list = cfg["targets"]
        self.targets = [t["mz"] for t in self.target_list]
        self.names = {t["mz"]: t.get("name", str(t["mz"])) for t in self.target_list}
        self.formulas = {t["mz"]: (t.get("formula_tex", ""), t.get("formula_plain", "")) for t in self.target_list}

        p = cfg.get("params", {})
        defaults = DEFAULT_TARGETS["params"]
        self.ntop = p.get("ntop", defaults["ntop"])
        self.halfwin = p.get("halfwin", defaults["halfwin"])
        self.grid = p.get("grid", defaults["grid"])
        self.ppm = p.get("ppm", defaults["ppm"])

    def mode_dir(self, mode):
        d = self.run_dir / mode
        if not d.is_dir():
            raise SystemExit(f"mode folder not found: {d}")
        return d

    def imzml_path(self, mode):
        d = self.mode_dir(mode)
        matches = sorted(d.glob("*.imzML")) or sorted(d.glob("*.imzml"))
        if not matches:
            raise SystemExit(f"no .imzML file found in {d}")
        if len(matches) > 1:
            raise SystemExit(f"multiple .imzML files found in {d}, expected exactly one: {matches}")
        return matches[0]

    def out(self, filename):
        return str(self.out_dir / filename)


def add_run_dir_args(parser, targets_default=True):
    parser.add_argument("run_dir", help="Path to a data-run folder (contains 'Profile Mode' / 'Centroid Mode' subfolders)")
    parser.add_argument("--out", default=None, help="Output folder (default: <run_dir>/output)")
    if targets_default:
        parser.add_argument("--targets", default=None, help="Path to a targets.json (default: <run_dir>/targets.json, else built-in defaults)")
    return parser
