"""Tests for the Part 1 fixes: case-insensitive .ibd resolution (a lowercase
.imzml used to silently break np.memmap's path), split profile/centroid
availability warnings in metrics.py, and parser/memmap handle closing.

Run with:
    python3 -m unittest discover -s tests -v
"""
import json
import math
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from imzml_fixtures import gaussian_profile_pixel, write_mode_folder  # noqa: E402

import run_pipeline  # noqa: E402
from msi_io import RunConfig, open_parser_and_ibd  # noqa: E402

TARGETS = [120.0234, 250.088]
PARAMS = dict(ntop=5, halfwin=0.03, grid=2e-4, ppm=200.0, intensity_floor=0.0)
GRID_XY = [(x, y) for x in range(1, 4) for y in range(1, 4)]  # 3x3 = 9 pixels


def _write_targets_json(run_dir, targets=TARGETS, params=PARAMS):
    payload = {
        "sample_name": "unit-test",
        "instrument_desc": "",
        "targets": [{"mz": t, "name": f"T{i}"} for i, t in enumerate(targets)],
        "params": params,
    }
    (run_dir / "targets.json").write_text(json.dumps(payload))


class LowercaseImzmlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_lowercase_imzml_extension_resolves_sibling_ibd(self):
        """A '.imzml' (lowercase) file must still find its '.ibd' sibling.
        The old `p.filename.replace(".imzML", ".ibd")` silently no-ops on a
        lowercase extension, so np.memmap ends up pointed at the .imzml XML
        file itself instead of the binary data."""
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        _write_targets_json(run_dir)
        pixels = [(x, y, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"])) for x, y in GRID_XY]
        imzml_path, ibd_path = write_mode_folder(run_dir / "Profile Mode", pixels, spectrum_type="profile")
        lower_path = imzml_path.with_suffix(".imzml")
        imzml_path.rename(lower_path)
        self.assertTrue(lower_path.is_file())
        self.assertTrue(ibd_path.is_file())

        cfg = RunConfig(run_dir)  # must resolve the lowercase file fine
        self.assertTrue(cfg.has_profile)

        run_pipeline.process_run(run_dir, None, None)  # must not raise

        with open(run_dir / "output" / "metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        for T in TARGETS:
            self.assertFalse(math.isnan(metrics[T]["apex"]), f"apex unexpectedly NaN for target {T}")
            self.assertAlmostEqual(metrics[T]["apex"], T, delta=PARAMS["grid"] * 2)


class ParserHandleCloseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_open_parser_and_ibd_closes_both_handles_on_exit(self):
        """Parser file handles used to never be closed, leaving the .imzML
        and .ibd files locked (notably on Windows). open_parser_and_ibd must
        close both the parser's own file object and the .ibd memmap."""
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        pixels = [(1, 1, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"]))]
        imzml_path, _ = write_mode_folder(run_dir / "Profile Mode", pixels, spectrum_type="profile")

        with open_parser_and_ibd(imzml_path) as (parser, buf):
            self.assertFalse(parser.m.closed)
            self.assertFalse(buf._mmap.closed)
        self.assertTrue(parser.m.closed)
        self.assertTrue(buf._mmap.closed)


class MetricsWarningTests(unittest.TestCase):
    """Exercises metrics.py's warning logic in isolation by crafting
    spectra.pkl directly, rather than round-tripping through pass1/pass2 -
    that lets each of the four mode/signal combinations below be set up
    exactly, instead of fighting the pixel-selection and floor logic to
    provoke them indirectly."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_distinct_mode_availability_vs_no_signal_warnings(self):
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        out_dir = run_dir / "output"
        out_dir.mkdir()

        halfwin, grid_step = 0.03, 2e-4
        T_both_clean = 100.0          # both modes present, real signal -> no warning
        T_no_profile_signal = 200.0   # Profile Mode present, but no signal above floor
        T_profile_absent = 300.0      # Profile Mode not part of this run at all
        T_centroid_absent = 400.0     # Centroid Mode not part of this run at all
        targets = [T_both_clean, T_no_profile_signal, T_profile_absent, T_centroid_absent]
        _write_targets_json(run_dir, targets=targets, params=dict(PARAMS, halfwin=halfwin, grid=grid_step))

        # RunConfig only needs *a* usable mode folder on disk to construct;
        # metrics.py reads spectra.pkl below, not this file's contents.
        stub_pixels = [(1, 1, *gaussian_profile_pixel([T_both_clean], halfwin, grid_step))]
        write_mode_folder(run_dir / "Profile Mode", stub_pixels, spectrum_type="profile")
        write_mode_folder(run_dir / "Centroid Mode", stub_pixels, spectrum_type="centroid")

        def gaussian(T):
            g, y = gaussian_profile_pixel([T], halfwin, grid_step)
            return g, y

        cases = [
            (T_both_clean, ("Profile Mode", "Centroid Mode"), True, True),
            (T_no_profile_signal, ("Profile Mode", "Centroid Mode"), False, False),
            (T_profile_absent, ("Centroid Mode",), False, False),
            (T_centroid_absent, ("Profile Mode",), True, False),
        ]
        spectra = {}
        for T, modes, prof_ok, cent_ok in cases:
            g, y = gaussian(T)
            if not prof_ok:
                y = np.zeros_like(g)
            cmz, cit, cn = (np.array([T]), np.array([800.0]), np.array([1])) if cent_ok \
                else (np.array([]), np.array([]), np.array([]))
            spectra[T] = dict(grid=g, prof=y, cmz=cmz, cit=cit, cn=cn,
                               nprof=(5 if prof_ok else 0), ncent=(5 if cent_ok else 0), modes=modes)
        with open(out_dir / "spectra.pkl", "wb") as f:
            pickle.dump(spectra, f)

        subprocess.run([sys.executable, str(_ROOT / "metrics.py"), str(run_dir)],
                        check=True, capture_output=True, text=True)

        with open(out_dir / "metrics.pkl", "rb") as f:
            metrics = pickle.load(f)

        self.assertEqual(metrics[T_both_clean]["warning"], "")

        w = metrics[T_no_profile_signal]["warning"]
        self.assertIn("no profile signal above intensity floor", w)
        self.assertNotIn("profile mode not available", w)

        w = metrics[T_profile_absent]["warning"]
        self.assertIn("profile mode not available", w)
        self.assertNotIn("no profile signal above intensity floor", w)

        w = metrics[T_centroid_absent]["warning"]
        self.assertIn("centroid mode not available", w)
        self.assertNotIn("no centroid peak detected near target", w)


if __name__ == "__main__":
    unittest.main()
