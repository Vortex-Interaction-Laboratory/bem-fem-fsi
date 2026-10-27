"""Tests of the compiled kernels (fast_kernels.py) against the vendored ones, standard library unittest.
python3 test_fast_kernels.py"""
import unittest

import numpy as np

import thick_panel_wing as tpw
import freewake_kernels as fk
import vent_mesh as vm
import fast_kernels as fkn


@unittest.skipUnless(fkn.ENABLED, "numba unavailable or BEM_NUMBA=0")
class TestAgainstVendored(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.pan = tpw.build_panels(vm.strut_mesh(n_c=8, nspan_half=6, h=1.0, alpha_deg=10.0, image=True,
                                                  tip_taper=0.2))

    def test_induced_velocity(self):
        r = self.rng
        pts = r.normal(size=(60, 3)); p1 = r.normal(size=(90, 3)); p2 = p1 + 0.2 * r.normal(size=(90, 3))
        g = r.normal(size=90); core = np.full(90, 0.05)
        pts[:3] = p1[:3]                                       # evaluation on segment end points
        a, b = fkn.induced_velocity(pts, p1, p2, g, core), fk.induced_velocity(pts, p1, p2, g, core)
        self.assertLess(np.abs(a - b).max(), 1e-12 * np.abs(b).max())

    def test_doublet_potential_on_and_off_the_panels(self):
        c = self.pan["corners"]
        pts = np.vstack([self.pan["centroids"], self.pan["centroids"] + 0.03 * self.rng.normal(size=self.pan["centroids"].shape)])
        a, b = fkn.doublet_potential_matrix(pts, c), tpw.doublet_potential_matrix(pts, c)
        # a panel's own centroid is an in-plane principal value (+-1/2 by the sign of a rounded zero), which the body
        # assembly overrides with -1/2 and the wake never meets; everything else agrees to round-off
        own = np.zeros(a.shape, dtype=bool); own[np.arange(len(c)), np.arange(len(c))] = True
        self.assertLess(np.abs(a - b)[~own].max(), 1e-12)

    def test_source_velocity_near_far_and_on_the_tip_line(self):
        sig = self.rng.normal(size=len(self.pan["areas"]))
        pts = np.vstack([self.pan["centroids"] + 0.02 * self.rng.normal(size=self.pan["centroids"].shape),
                         self.pan["points"][:, 0], 30.0 * self.rng.normal(size=(20, 3))])
        a, b = fkn.source_velocity(pts, self.pan["corners"], sig), tpw.source_velocity(pts, self.pan["corners"], sig)
        self.assertLess(np.abs(a - b).max(), 1e-12 * np.abs(b).max())

    def test_wake_influence(self):
        onset = fk.make_onset(np.array([-10.0, 10.0]), np.array([1.0, 1.0]))
        pts = vm.strut_mesh(n_c=8, nspan_half=6, h=1.0, alpha_deg=10.0, image=True)
        st = tpw.init_state(pts, onset, 1.0, config={"lifting": True, "nwake": 6})
        st["wake"]["mu"][:] = self.rng.normal(size=st["wake"]["mu"].shape)
        for n_b in (6, 3):                                     # all rows bound, and a frozen tail
            st["wake"]["n_bound"] = n_b
            a = fkn.wake_influence(st["panels"]["centroids"], st["wake"], onset)
            b = tpw.wake_influence(st["panels"]["centroids"], st["wake"], onset)
            for x, y in zip(a, b):
                self.assertLess(np.abs(x - y).max(), 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
