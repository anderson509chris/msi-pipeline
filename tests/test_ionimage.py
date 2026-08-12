"""Tests for ionimage.py: grid construction (unsampled pixels stay NaN, not
zero), vmax percentile clipping (a single hot pixel must not set the colour
scale), and the CLI end to end (PNG outputs, graceful --roi degradation when
spectra.pkl / common_pixels.pkl don't exist yet).

Run with:
    python3 -m unittest discover -s tests -v
"""
import json
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

import ionimage  # noqa: E402

TARGETS = [120.0234, 250.088]
PARAMS = dict(ntop=5, halfwin=0.03, grid=2e-4, ppm=200.0, intensity_floor=0.0)


def _write_targets_json(run_dir, targets=TARGETS, params=PARAMS):
    payload = {
        "sample_name": "ionimage-test",
        "instrument_desc": "",
        "targets": [{"mz": t, "name": f"T{i}"} for i, t in enumerate(targets)],
        "params": params,
    }
    (run_dir / "targets.json").write_text(json.dumps(payload))


class BuildGridTests(unittest.TestCase):
    def test_gap_pixel_is_nan_not_zero(self):
        """A grid position with no entry in coords must stay NaN - a real
        zero-intensity pixel and an unsampled position must not look the
        same on the rendered image."""
        # 4 (x) wide x 3 (y) tall grid, every position sampled except (2, 2)
        coords = [(x, y) for y in range(1, 4) for x in range(1, 5) if (x, y) != (2, 2)]
        values = np.arange(1, len(coords) + 1, dtype=np.float64)

        grid = ionimage.build_grid(coords, values)

        self.assertEqual(grid.shape, (3, 4))
        self.assertTrue(np.isnan(grid[1, 1]))  # (x=2, y=2) -> grid[y-1, x-1]
        self.assertEqual(np.isnan(grid).sum(), 1)
        for (x, y), v in zip(coords, values):
            self.assertEqual(grid[y - 1, x - 1], v)


class ComputeVmaxTests(unittest.TestCase):
    def test_hot_pixel_does_not_set_vmax(self):
        """One extreme pixel must not become the colour-scale max - vmax
        should be the 99th percentile of above-floor pixels, well below the
        hot pixel's own value, or a single hot pixel would flatten the rest
        of the image into a uniform dark rectangle."""
        rng = np.random.default_rng(0)
        grid = rng.uniform(50, 150, size=(20, 20))
        grid[5, 5] = 1_000_000.0  # hot pixel

        vmax = ionimage.compute_vmax(grid, floor=0.0, percentile=99.0)

        self.assertLess(vmax, 1000.0)
        self.assertGreater(vmax, 50.0)

    def test_respects_intensity_floor(self):
        """Values at/below the floor must not count toward the percentile,
        mirroring what pixel selection treats as noise elsewhere."""
        grid = np.array([[0.0, 0.0], [100.0, 200.0]])
        vmax_floor0 = ionimage.compute_vmax(grid, floor=0.0, percentile=50.0)
        vmax_floor150 = ionimage.compute_vmax(grid, floor=150.0, percentile=50.0)
        self.assertEqual(vmax_floor0, 150.0)   # median of [100, 200]
        self.assertEqual(vmax_floor150, 200.0)  # only 200 clears the floor


class IonImageCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run_dir_with_pass1_output(self):
        """A run folder with a minimal-but-valid Profile Mode folder (so
        RunConfig accepts it) and hand-crafted pass1 output - peak_prof.npy /
        coords_prof.pkl with a deliberate gap pixel and a deliberate hot
        pixel - so ionimage.py's CLI is exercised exactly as it would run
        standalone right after pass1, without a full imzML round-trip."""
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        _write_targets_json(run_dir)
        stub_pixels = [(1, 1, *gaussian_profile_pixel(TARGETS, PARAMS["halfwin"], PARAMS["grid"]))]
        write_mode_folder(run_dir / "Profile Mode", stub_pixels, spectrum_type="profile")

        out_dir = run_dir / "output"
        out_dir.mkdir()
        # 12x12 grid (144 positions), all but one sampled - enough pixels for
        # a 99th-percentile clip to be statistically meaningful (with only a
        # handful of pixels, the top percentile lands on the hot pixel itself
        # by construction, which isn't representative of real MSI data).
        coords = [(x, y) for y in range(1, 13) for x in range(1, 13) if (x, y) != (2, 2)]  # gap at (2, 2)
        n = len(coords)
        rng = np.random.default_rng(1)
        peak = rng.uniform(50, 150, size=(len(TARGETS), n))
        peak[:, 0] = 1_000_000.0  # hot pixel, shared position for both targets
        np.save(out_dir / "peak_prof.npy", peak)
        with open(out_dir / "coords_prof.pkl", "wb") as f:
            pickle.dump(coords, f)
        return run_dir, out_dir

    def test_png_created_for_each_target_and_combined_figure(self):
        run_dir, out_dir = self._run_dir_with_pass1_output()

        result = subprocess.run(
            [sys.executable, str(_ROOT / "ionimage.py"), str(run_dir), "--roi", "none"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        for T in TARGETS:
            self.assertTrue((out_dir / f"ion_image_mz{T:.4f}.png").is_file())
        self.assertTrue((out_dir / "Fig_S2_ion_images.png").is_file())
        self.assertTrue((out_dir / "Fig_S2_ion_images.pdf").is_file())

    def test_roi_none_works_when_spectra_pkl_absent(self):
        """--roi none must work standalone right after pass1, before
        pass2/common have produced spectra.pkl / common_pixels.pkl."""
        run_dir, out_dir = self._run_dir_with_pass1_output()
        self.assertFalse((out_dir / "spectra.pkl").exists())
        self.assertFalse((out_dir / "common_pixels.pkl").exists())

        result = subprocess.run(
            [sys.executable, str(_ROOT / "ionimage.py"), str(run_dir), "--roi", "none"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_roi_degrades_gracefully_when_spectra_pkl_absent(self):
        """The default --roi per-target must warn and skip the overlay
        instead of crashing when spectra.pkl doesn't exist yet."""
        run_dir, out_dir = self._run_dir_with_pass1_output()

        result = subprocess.run(
            [sys.executable, str(_ROOT / "ionimage.py"), str(run_dir)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WARNING", result.stdout)
        for T in TARGETS:
            self.assertTrue((out_dir / f"ion_image_mz{T:.4f}.png").is_file())

    def test_gap_pixel_stays_nan_through_cli_generated_data(self):
        """Cross-check build_grid's unit behaviour against the exact coords
        the CLI-level fixture above writes to disk."""
        _run_dir, out_dir = self._run_dir_with_pass1_output()
        peak = np.load(out_dir / "peak_prof.npy")
        with open(out_dir / "coords_prof.pkl", "rb") as f:
            coords = pickle.load(f)

        grid = ionimage.build_grid(coords, peak[0])

        self.assertTrue(np.isnan(grid[1, 1]))  # (x=2, y=2) never in coords


if __name__ == "__main__":
    unittest.main()
