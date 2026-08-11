"""Tests for single-acquisition-mode run folders (Profile-Mode-only,
Centroid-Mode-only), broken-mode-folder detection, and mzPrecision handling.

Run with:
    python3 -m unittest discover -s tests -v
from the project root (needs the project's own .venv active - no new
dependencies are used beyond what the project already requires).
"""
import json
import math
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from imzml_fixtures import centroid_pixel, gaussian_profile_pixel, write_mode_folder  # noqa: E402

import run_pipeline  # noqa: E402
from msi_io import RunConfig, RunConfigError  # noqa: E402

TARGETS = [120.0234, 250.088]
PARAMS = dict(ntop=5, halfwin=0.03, grid=2e-4, ppm=200.0, intensity_floor=0.0)
GRID_XY = [(x, y) for x in range(1, 4) for y in range(1, 4)]  # 3x3 = 9 pixels


def _write_targets_json(run_dir):
    payload = {
        "sample_name": "unit-test",
        "instrument_desc": "",
        "targets": [{"mz": t, "name": f"T{i}"} for i, t in enumerate(TARGETS)],
        "params": PARAMS,
    }
    (run_dir / "targets.json").write_text(json.dumps(payload))


def _write_profile_mode(run_dir, precision="f", mode_dir_name="Profile Mode"):
    pixels = [(x, y, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"])) for x, y in GRID_XY]
    return write_mode_folder(run_dir / mode_dir_name, pixels, mz_precision=precision,
                              intensity_precision=precision, spectrum_type="profile")


def _write_centroid_mode(run_dir, precision="f", mode_dir_name="Centroid Mode"):
    pixels = [(x, y, *centroid_pixel(TARGETS)) for x, y in GRID_XY]
    return write_mode_folder(run_dir / mode_dir_name, pixels, mz_precision=precision,
                              intensity_precision=precision, spectrum_type="centroid")


def _load_metrics(run_dir):
    with open(run_dir / "output" / "metrics.pkl", "rb") as f:
        return pickle.load(f)


class SingleModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run_dir(self, name):
        d = self.tmp / name
        d.mkdir()
        _write_targets_json(d)
        return d

    def test_both_modes_completes_with_no_nan(self):
        """Both modes present ('f' precision): pipeline completes and every
        metric is a real number - the full-data path is unaffected."""
        run_dir = self._run_dir("both")
        _write_profile_mode(run_dir, precision="f")
        _write_centroid_mode(run_dir, precision="f")

        run_pipeline.process_run(run_dir, None, None)

        metrics = _load_metrics(run_dir)
        for T in TARGETS:
            m = metrics[T]
            self.assertEqual(m["modes"], ("Profile Mode", "Centroid Mode"))
            self.assertEqual(m["warning"], "")
            for field in ("apex", "ymax", "fwhm", "R", "pcen", "ppm_prof", "cmz", "cit", "ppm_cent", "areafrac"):
                self.assertFalse(math.isnan(m[field]), f"{field} unexpectedly NaN for target {T}")
            self.assertAlmostEqual(m["apex"], T, delta=PARAMS["grid"] * 2)
            self.assertAlmostEqual(m["cmz"], T, delta=5e-5)  # centroid mz written as float32 here

    def test_profile_only_leaves_centroid_fields_nan(self):
        """Profile Mode only ('d' precision, exercising 64-bit reads end to
        end): profile-side metrics are real numbers, centroid-side metrics
        are NaN with the expected warning - never a crash, never a silently
        wrong value."""
        run_dir = self._run_dir("profile_only")
        _write_profile_mode(run_dir, precision="d")

        run_pipeline.process_run(run_dir, None, None)

        metrics = _load_metrics(run_dir)
        for T in TARGETS:
            m = metrics[T]
            self.assertEqual(m["modes"], ("Profile Mode",))
            self.assertIn("no centroid peak detected near target", m["warning"])
            for field in ("apex", "ymax", "fwhm", "R", "pcen", "ppm_prof"):
                self.assertFalse(math.isnan(m[field]), f"{field} unexpectedly NaN for target {T}")
            self.assertAlmostEqual(m["apex"], T, delta=PARAMS["grid"] * 2)
            for field in ("cmz", "cit", "ppm_cent"):
                self.assertTrue(math.isnan(m[field]), f"{field} expected NaN for target {T}")

    def test_centroid_only_leaves_profile_fields_nan(self):
        """Centroid Mode only ('f' precision): centroid-side metrics are real
        numbers, profile-side metrics are NaN with the mirror warning."""
        run_dir = self._run_dir("centroid_only")
        _write_centroid_mode(run_dir, precision="f")

        run_pipeline.process_run(run_dir, None, None)

        metrics = _load_metrics(run_dir)
        for T in TARGETS:
            m = metrics[T]
            self.assertEqual(m["modes"], ("Centroid Mode",))
            self.assertIn("profile mode not available", m["warning"])
            for field in ("apex", "ymax", "fwhm", "R", "pcen", "ppm_prof", "areafrac"):
                self.assertTrue(math.isnan(m[field]), f"{field} expected NaN for target {T}")
            self.assertFalse(math.isnan(m["cmz"]))
            self.assertAlmostEqual(m["cmz"], T, delta=5e-5)  # centroid mz written as float32 here

    def test_broken_mode_folder_raises_instead_of_degrading(self):
        """A present Profile Mode folder with two .imzML files must raise -
        it must NOT silently fall back to Centroid-Mode-only, even though
        Centroid Mode is otherwise perfectly usable."""
        run_dir = self._run_dir("broken")
        _write_centroid_mode(run_dir, precision="f")
        prof_dir = run_dir / "Profile Mode"
        prof_dir.mkdir()
        pixels = [(x, y, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"])) for x, y in GRID_XY]
        write_mode_folder(prof_dir, pixels, stem="a")
        write_mode_folder(prof_dir, pixels, stem="b")  # second .imzML -> ambiguous/broken

        with self.assertRaises(RunConfigError):
            RunConfig(run_dir)

        with self.assertRaises(RunConfigError):
            run_pipeline.process_run(run_dir, None, None)

    def test_appledouble_companion_file_ignored(self):
        """A hidden "._sample.imzML" AppleDouble companion file (which macOS
        creates automatically when data is copied from a USB drive, external
        disk, or network share) must not be counted as a second, ambiguous
        .imzML file - pathlib's "*" glob matches dotfiles even though a shell
        glob wouldn't, so this is a real trap, not a hypothetical one."""
        run_dir = self._run_dir("appledouble")
        _write_profile_mode(run_dir, precision="f")
        imzml_path = next((run_dir / "Profile Mode").glob("*.imzML"))
        (imzml_path.parent / f"._{imzml_path.name}").write_bytes(b"\x00\x05\x16\x07")

        cfg = RunConfig(run_dir)  # must not raise
        self.assertTrue(cfg.has_profile)
        self.assertEqual(cfg.modes, ("Profile Mode",))

    def test_flat_layout_profile_auto_detected(self):
        """No 'Profile Mode'/'Centroid Mode' subfolder at all - just a single
        .imzML + .ibd pair directly in the run folder. The mode must be
        auto-detected from the file's own declared spectrum type and the
        pipeline must run end to end exactly as it would from a properly
        named subfolder."""
        run_dir = self._run_dir("flat_profile")
        _write_profile_mode(run_dir, precision="f", mode_dir_name=".")

        cfg = RunConfig(run_dir)
        self.assertTrue(cfg.has_profile)
        self.assertFalse(cfg.has_centroid)
        self.assertEqual(cfg.modes, ("Profile Mode",))

        run_pipeline.process_run(run_dir, None, None)
        metrics = _load_metrics(run_dir)
        for T in TARGETS:
            m = metrics[T]
            self.assertFalse(math.isnan(m["apex"]))
            self.assertAlmostEqual(m["apex"], T, delta=PARAMS["grid"] * 2)
            self.assertTrue(math.isnan(m["cmz"]))

    def test_flat_layout_centroid_auto_detected(self):
        run_dir = self._run_dir("flat_centroid")
        _write_centroid_mode(run_dir, precision="f", mode_dir_name=".")

        cfg = RunConfig(run_dir)
        self.assertTrue(cfg.has_centroid)
        self.assertFalse(cfg.has_profile)
        self.assertEqual(cfg.modes, ("Centroid Mode",))

    def test_flat_layout_undeclared_type_raises(self):
        """A flat .imzML that doesn't declare profile or centroid spectrum
        type must raise with a clear explanation, not silently guess."""
        run_dir = self._run_dir("flat_undeclared")
        pixels = [(x, y, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"])) for x, y in GRID_XY]
        write_mode_folder(run_dir, pixels)  # spectrum_type=None -> undeclared

        with self.assertRaises(RunConfigError):
            RunConfig(run_dir)

    def test_broken_mode_folder_missing_ibd_raises(self):
        """A Profile Mode folder with an .imzML but no matching .ibd must
        raise, not be treated as absent."""
        run_dir = self._run_dir("missing_ibd")
        _write_centroid_mode(run_dir, precision="f")
        prof_dir = run_dir / "Profile Mode"
        prof_dir.mkdir()
        pixels = [(x, y, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"])) for x, y in GRID_XY]
        _, ibd_path = write_mode_folder(prof_dir, pixels)
        ibd_path.unlink()

        with self.assertRaises(RunConfigError):
            RunConfig(run_dir)

    def test_no_usable_mode_raises(self):
        run_dir = self.tmp / "empty"
        run_dir.mkdir()
        _write_targets_json(run_dir)
        with self.assertRaises(RunConfigError):
            RunConfig(run_dir)


if __name__ == "__main__":
    unittest.main()
