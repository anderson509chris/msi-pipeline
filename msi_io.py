"""Shared config/IO helpers so each pipeline stage can be pointed at any run folder.

A "run folder" is expected to look like:

    <run_dir>/
        Profile Mode/<name>.imzML (+ .ibd)     \\_ at least one of these two
        Centroid Mode/<name>.imzML (+ .ibd)    /   must be present and usable
        targets.json        (optional, see DEFAULT_TARGETS below)

Alternatively, if neither subfolder exists, <run_dir> may directly contain a
single .imzML + .ibd pair with no wrapper subfolder at all - the acquisition
mode is then auto-detected from the file's own declared spectrum
representation (profile vs centroid), not from folder naming.

All intermediate/output artifacts (peak_*.npy, spectra*.pkl, metrics*.pkl,
figures, CSVs) are written to <run_dir>/output/ by default, or to --out if given.
"""
import json
import os
from pathlib import Path

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser

MODES = ("Profile Mode", "Centroid Mode")


class RunConfigError(Exception):
    """Raised for anything wrong with a run folder or its targets config.
    Callers (CLI entry points, the Streamlit GUI) catch this to show a clean
    message instead of a stack trace / hard process exit."""


def read_binary_array(buf, offset, n_elements, precision):
    """Read n_elements starting at offset (bytes) from an .ibd memmap/buffer,
    using the array's declared imzML precision ('f' = 32-bit, 'd' = 64-bit float).
    Reading as the wrong width silently returns garbage, so always pass
    p.mzPrecision / p.intensityPrecision here rather than assuming float32."""
    itemsize = np.dtype(precision).itemsize
    return buf[offset:offset + n_elements * itemsize].view(precision)


def _find_imzml_files(d):
    """.imzML files directly in d, case-insensitively, deduplicated by resolved
    path (globbing both *.imzML and *.imzml double-counts on case-insensitive
    filesystems otherwise). Skips hidden dotfiles - pathlib's "*" glob matches
    them even though shell globs don't, and macOS silently creates hidden
    "._name.imzML" AppleDouble companion files when data is copied from a
    USB drive, external disk, or network share, which would otherwise look
    like a second, ambiguous .imzML file and wrongly flag the folder as broken."""
    matches = {f.resolve() for f in d.glob("*.imzML") if not f.name.startswith(".")} \
        | {f.resolve() for f in d.glob("*.imzml") if not f.name.startswith(".")}
    return sorted(matches)


def _find_sibling_ibd(imzml_path):
    """The .ibd file sharing imzml_path's stem, matched case-insensitively on
    the extension since acquisition software isn't consistent about casing."""
    for f in imzml_path.parent.iterdir():
        if f.is_file() and f.stem == imzml_path.stem and f.suffix.lower() == ".ibd":
            return f
    return None


def mode_status(mode_dir):
    """Classify one acquisition-mode folder (e.g. <run>/Profile Mode):
      ("absent", None)       - folder doesn't exist: fine, degrade to the other mode
      ("broken", reason)     - folder exists but isn't usable: caller must raise, not degrade
      ("ok", imzml_path)     - exactly one .imzML with a matching .ibd
    Shared by RunConfig (which raises on "broken") and run_pipeline.has_modes
    (which only needs the ok/not-ok distinction) so the two can't drift apart."""
    if not mode_dir.is_dir():
        return "absent", None
    imzmls = _find_imzml_files(mode_dir)
    if not imzmls:
        return "broken", f"{mode_dir} exists but contains no .imzML file"
    if len(imzmls) > 1:
        return "broken", f"{mode_dir} contains multiple .imzML files, expected exactly one: {imzmls}"
    ibd = _find_sibling_ibd(imzmls[0])
    if ibd is None:
        return "broken", f"{imzmls[0]} has no matching .ibd file (expected {imzmls[0].with_suffix('.ibd')})"
    return "ok", imzmls[0]


# Fallback target list, used only if a run folder has no targets.json.
DEFAULT_TARGETS = {
    "sample_name": "EpCtrl-4-1_2_S2_SM_Neg_20240306_IT",
    "instrument_desc": "Orbitrap MALDI-MSI, negative ion mode, m/z 70–500  ·  142 × 308 px, 20 µm  ·  lock mass 157.07712",
    "targets": [
        {"mz": 140.0118, "name": "Phosphoethanolamine", "formula_plain": "C2H8NO4P"},
        {"mz": 146.0459, "name": "Glutamate", "formula_plain": "C5H9NO4"},
        {"mz": 151.0261, "name": "Xanthine", "formula_plain": "C5H4N4O2"},
        {"mz": 215.0328, "name": "Glucose", "formula_plain": "C5H13O7P"},
    ],
    "params": {"ntop": 100, "halfwin": 0.06, "grid": 1e-5, "ppm": 3.0, "intensity_floor": 0.0},
}


class RunConfig:
    def __init__(self, run_dir, out_dir=None, targets_path=None):
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise RunConfigError(f"run folder not found: {self.run_dir}")
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
        self.formulas = {t["mz"]: t.get("formula_plain", "") for t in self.target_list}

        p = cfg.get("params", {})
        defaults = DEFAULT_TARGETS["params"]
        self.ntop = p.get("ntop", defaults["ntop"])
        self.halfwin = p.get("halfwin", defaults["halfwin"])
        self.grid = p.get("grid", defaults["grid"])
        self.ppm = p.get("ppm", defaults["ppm"])
        self.intensity_floor = p.get("intensity_floor", defaults["intensity_floor"])

        self._detect_modes()

    def _detect_modes(self):
        """A mode folder is either absent (fine, we degrade to the other mode)
        or present-and-broken (not fine - raise, never silently drop data).
        Only an absent folder is allowed to mean 'not usable'."""
        self._mode_paths = {}
        broken = []
        for mode in MODES:
            status, info = mode_status(self.run_dir / mode)
            if status == "ok":
                self._mode_paths[mode] = info
            elif status == "broken":
                broken.append(info)

        if broken:
            raise RunConfigError("; ".join(broken))

        if not self._mode_paths:
            # Neither 'Profile Mode' nor 'Centroid Mode' subfolder exists at
            # all - fall back to a flat layout where run_dir itself directly
            # holds one .imzML + .ibd pair, mode auto-detected from content.
            status, info = mode_status(self.run_dir)
            if status == "broken":
                raise RunConfigError(info)
            if status == "ok":
                mode = self._sniff_mode(info)
                if mode is None:
                    raise RunConfigError(
                        f"{info} doesn't declare whether it's profile or centroid data "
                        f"(no 'profile spectrum' / 'centroid spectrum' cvParam found) - "
                        f"put it in a 'Profile Mode' or 'Centroid Mode' subfolder instead"
                    )
                self._mode_paths[mode] = info

        if not self._mode_paths:
            raise RunConfigError(
                f"no usable acquisition-mode data found in {self.run_dir} "
                f"(expected a 'Profile Mode' and/or 'Centroid Mode' subfolder, "
                f"each with exactly one .imzML file and a matching .ibd; or a single "
                f".imzML + .ibd pair directly in the folder)"
            )
        self.modes = tuple(m for m in MODES if m in self._mode_paths)
        self.has_profile = "Profile Mode" in self._mode_paths
        self.has_centroid = "Centroid Mode" in self._mode_paths

    @staticmethod
    def _sniff_mode(imzml_path):
        """Auto-detect profile vs centroid from the file's own declared
        spectrum representation, for a flat-layout .imzML with no
        'Profile Mode'/'Centroid Mode' wrapper folder to name it. Returns
        None if the file doesn't declare one clearly (caller then raises,
        rather than guessing)."""
        try:
            p = ImzMLParser(str(imzml_path))
        except Exception as e:
            raise RunConfigError(f"could not read {imzml_path}: {e}")
        try:
            if p.spectrum_mode == "profile":
                return "Profile Mode"
            if p.spectrum_mode == "centroid":
                return "Centroid Mode"
            return None
        finally:
            if p.m is not None:
                p.m.close()

    def mode_banner(self):
        if self.has_profile and self.has_centroid:
            return "Profile Mode and Centroid Mode found. All metrics available."
        if self.has_profile:
            return ("Profile Mode only - no Centroid Mode folder found in this run. "
                    "Peak shape metrics (FWHM, resolving power, profile mass error) are available; "
                    "centroid mass error, stick overlay, and peak-count-in-window are not.")
        return ("Centroid Mode only - no Profile Mode folder found in this run. "
                "Centroid mass error and stick overlay are available; "
                "FWHM, resolving power, profile mass error, and area fraction are not "
                "(these require the profile peak shape).")

    def mode_dir(self, mode):
        if mode not in self._mode_paths:
            raise RunConfigError(f"mode not available in this run: {mode}")
        return self._mode_paths[mode].parent

    def imzml_path(self, mode):
        if mode not in self._mode_paths:
            raise RunConfigError(f"mode not available in this run: {mode}")
        return self._mode_paths[mode]

    def out(self, filename):
        return str(self.out_dir / filename)


def add_run_dir_args(parser, targets_default=True):
    parser.add_argument("run_dir", help="Path to a data-run folder (a usable 'Profile Mode' and/or 'Centroid Mode' subfolder)")
    parser.add_argument("--out", default=None, help="Output folder (default: <run_dir>/output)")
    if targets_default:
        parser.add_argument("--targets", default=None, help="Path to a targets.json (default: <run_dir>/targets.json, else built-in defaults)")
    return parser
