"""Tests of tip shedding (tip_wake.py), standard library unittest. python3 test_tip_wake.py"""
import unittest
import warnings

import numpy as np

import thick_panel_wing as tpw
import freewake_kernels as fk
import vent_cavity as vcv
import vent_mesh as vm
import vent_solve as vsl
import tip_wake as tw


def strut(tip, regime="FW", n_c=8, nspan_half=6, alpha=10.0):
    points = vm.strut_mesh(n_c=n_c, nspan_half=nspan_half, h=1.0, chord=1.0,
                           alpha_deg=alpha, image=True)
    maps = vm.mirror_maps(points.shape, image=True)
    half = vm.half_points(points, maps)
    onset = fk.make_onset(np.array([-10.0, 10.0]), np.array([1.0, 1.0]))
    fluid = tpw.init_state(points, onset, 1.0, config={"lifting": True, "nwake": 6})
    if tip:
        tw.add_tip_wakes(fluid)
    vent = vcv.build_vent(maps, half, h=1.0, chord=1.0, u_ref=1.0,
                          alpha_rad=np.radians(alpha), rho=1000.0, fn_h=2.5,
                          regime=regime)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fluid, cav, loads = vsl.solve_cavity(fluid, vent, rho=1000.0)
    return fluid, loads


class TestTipEdge(unittest.TestCase):

    def test_the_tip_line_is_the_pinched_camber_line(self):
        points = vm.strut_mesh(n_c=6, nspan_half=4, h=1.0, chord=1.0, alpha_deg=5.0)
        pan = tpw.build_panels(points)
        for side in (0, -1):
            nodes, k_up, k_low = tw.tip_edge(points, pan, side)
            self.assertEqual(nodes.shape, (7, 3))
            self.assertEqual(len(k_up), 6)
            j = 0 if side == 0 else pan["nspan"] - 1
            self.assertTrue(np.all(k_up % pan["nspan"] == j))
            self.assertTrue(np.all(k_low % pan["nspan"] == j))
            # the strip next to the trailing edge is the trailing-edge Kutta pair
            self.assertEqual(k_up[-1 if side == 0 else 0], pan["k_up"][j])
            self.assertEqual(k_low[-1 if side == 0 else 0], pan["k_low"][j])

    def test_an_open_station_is_refused(self):
        points = vm.strut_mesh(n_c=6, nspan_half=4, h=1.0, chord=1.0, alpha_deg=5.0)
        pan = tpw.build_panels(points)
        opened = points.copy()
        opened[:6, 0, 2] -= 0.01                    # unpinch the lower surface
        with self.assertRaises(ValueError):
            tw.tip_edge(opened, pan, 0)


class TestTipFold(unittest.TestCase):

    def test_without_sheets_the_fold_is_unchanged(self):
        fluid, _ = strut(False)
        pan = fluid["panels"]
        pb, pk = tpw.wake_influence(pan["centroids"], fluid["wake"], fluid["onset"])
        out = tw.fold_arrays(fluid, pan["centroids"], pb, pk, pan["k_up"], pan["k_low"])
        self.assertIs(out[0], pb)
        self.assertIs(out[2], pan["k_up"])

    def test_the_sheet_meets_the_trailing_edge_wake_at_the_corner(self):
        fluid, _ = strut(True)
        for t in fluid["tip_wakes"]:
            mu_tip = t["wake"]["mu"][0]
            corner = mu_tip[-1] if t["side"] == 0 else mu_tip[0]
            j = 0 if t["side"] == 0 else fluid["panels"]["nspan"] - 1
            self.assertAlmostEqual(corner, fluid["wake"]["mu"][0, j], places=12)

    def test_the_sheet_carries_the_cumulative_tip_loading(self):
        # leading edge to trailing edge the jump grows, as the circulation of the
        # VLM tip rings does; the flipped orientation gives a negative jump and less lift
        fluid, loads = strut(True)
        mu = fluid["tip_wakes"][0]["wake"]["mu"][0]
        self.assertTrue(np.all(mu > 0.0))
        self.assertTrue(np.all(np.diff(mu) > -1e-3 * mu.max()))

    def test_tip_shedding_raises_the_lift_wetted_and_ventilated(self):
        for regime in ("FW", "FV"):
            _, a = strut(False, regime)
            _, b = strut(True, regime)
            self.assertGreater(b["resultants"]["CL"], 1.05 * a["resultants"]["CL"])

    def test_the_relaxed_wake_refuses_tip_sheets(self):
        fluid, _ = strut(True)
        points = fluid["panels"]["points"]
        maps = vm.mirror_maps(points.shape, image=True)
        vent = vcv.build_vent(maps, vm.half_points(points, maps), h=1.0, chord=1.0,
                              u_ref=1.0, alpha_rad=np.radians(10.0), rho=1000.0,
                              fn_h=2.5, regime="FW")
        with self.assertRaises(NotImplementedError):
            vsl.relax_wake(fluid, vent)


class TestTaper(unittest.TestCase):

    def test_the_taper_thins_the_tip_linearly_and_keeps_the_image(self):
        p = vm.strut_mesh(n_c=6, nspan_half=8, h=1.0, alpha_deg=0.0, tip_taper=0.25)
        t = np.ptp(p[:, :9, 2], axis=0)
        self.assertEqual(t[0], 0.0)
        self.assertAlmostEqual(t[1] / t[3], 0.5, places=12)
        self.assertEqual(t[2], t[3])
        self.assertEqual(vm.symmetry_residual(p), 0.0)

    def test_no_taper_is_the_default_mesh(self):
        a = vm.strut_mesh(n_c=6, nspan_half=8, h=1.0, alpha_deg=5.0)
        b = vm.strut_mesh(n_c=6, nspan_half=8, h=1.0, alpha_deg=5.0, tip_taper=0.0)
        self.assertTrue(np.array_equal(a, b))


class TestSpacing(unittest.TestCase):

    def test_the_spacings_keep_the_strut_exact(self):
        for sp in ("uniform", "tip", "cosine"):
            p = vm.strut_mesh(n_c=6, nspan_half=6, h=1.0, alpha_deg=5.0, spacing=sp)
            self.assertEqual(vm.symmetry_residual(p), 0.0)
            y = p[3, :7, 1]
            self.assertAlmostEqual(y[0], -1.0, places=14)
            self.assertAlmostEqual(y[-1], 0.0, places=14)
            self.assertTrue(np.all(np.diff(y) > 0.0))

    def test_cosine_clusters_at_the_tip_and_the_waterline(self):
        p = vm.strut_mesh(n_c=6, nspan_half=6, h=1.0, alpha_deg=5.0, spacing="cosine")
        dy = np.diff(p[3, :7, 1])
        self.assertLess(dy[0], dy[3]); self.assertLess(dy[-1], dy[3])

    def test_an_unknown_spacing_is_refused(self):
        with self.assertRaises(ValueError):
            vm.strut_mesh(n_c=6, nspan_half=6, spacing="sine")


if __name__ == "__main__":
    unittest.main(verbosity=2)
