"""Compiled (numba) versions of the panel-method kernels the wake relaxation spends its time in.

A relaxed solve with tip sheets spends nearly all of each sweep in three vendored numpy kernels: the Biot-Savart
velocity of the vortex segments at the wake nodes (freewake_kernels.induced_velocity), the velocity of the body
sources there (thick_panel_wing.source_velocity) and the doublet potential of the wake quads at the collocation
points (thick_panel_wing.doublet_potential_matrix, inside wake_influence). Both
build large temporary arrays; compiled loops do the same arithmetic without them. The vendored code is not
touched: this module re-implements the same formulas, and the modules written here (tip_wake, vent_solve) call
these versions when numba is available.

Four sections, and the same three discipline rules: nothing imports downward, every block is a pure function over
explicit arguments, and prompts and prints appear only under main().

    1 imports       numpy, numba (optional); thick_panel_wing for the wake geometry
    2 kernels       segment velocities, panel solid angles, Hess source velocity
    3 drop-ins      source_velocity, induced_velocity, doublet_potential_matrix, wake_influence, with the vendored
                    signatures
    4 main          agreement and speed against the vendored kernels

**Same formulas, same order of operations.** The segment kernel is the van Garrel-regularised Biot-Savart law of
freewake_kernels.segment_velocities, with the same floors (1e-12 on the distances, 1e-30 on the denominator); the
solid angle is the van Oosterom-Strackee formula of thick_panel_wing._solid_angle_tri averaged over the same four
triangles. No fastmath: the results agree with the vendored kernels to round-off (test_fast_kernels.py checks
1e-12 relative), so switching them changes no result beyond round-off.

**Off switch.** BEM_NUMBA=0 in the environment, or numba missing, makes every drop-in call the vendored function.
"""

# --------------------------------------------------------------- 1 imports

import os

import numpy as np

import thick_panel_wing as tpw
import freewake_kernels as fk

try:
    import numba as nb
except ImportError:                     # pragma: no cover - numba is optional
    nb = None

ENABLED = nb is not None and os.environ.get("BEM_NUMBA", "1") != "0"


# --------------------------------------------------------------- 2 kernels

if nb is not None:

    @nb.njit(parallel=True, cache=True)
    def _induced(points, p1, p2, g, core):
        m, s = points.shape[0], p1.shape[0]
        out = np.zeros((m, 3))
        for i in nb.prange(m):
            px, py, pz = points[i, 0], points[i, 1], points[i, 2]
            vx = vy = vz = 0.0
            for j in range(s):
                r1x, r1y, r1z = px - p1[j, 0], py - p1[j, 1], pz - p1[j, 2]
                r2x, r2y, r2z = px - p2[j, 0], py - p2[j, 1], pz - p2[j, 2]
                r0x, r0y, r0z = p2[j, 0] - p1[j, 0], p2[j, 1] - p1[j, 1], p2[j, 2] - p1[j, 2]
                cx = r1y * r2z - r1z * r2y
                cy = r1z * r2x - r1x * r2z
                cz = r1x * r2y - r1y * r2x
                cr2 = cx * cx + cy * cy + cz * cz
                r0n2 = r0x * r0x + r0y * r0y + r0z * r0z
                r1n = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
                r2n = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
                dot1 = r0x * r1x + r0y * r1y + r0z * r1z
                dot2 = r0x * r2x + r0y * r2y + r0z * r2z
                denom = 4.0 * np.pi * (cr2 + core[j] * core[j] * r0n2)
                k = dot1 / max(r1n, 1e-12) - dot2 / max(r2n, 1e-12)
                k /= max(denom, 1e-30)
                k *= g[j]
                vx += k * cx; vy += k * cy; vz += k * cz
            out[i, 0] = vx; out[i, 1] = vy; out[i, 2] = vz
        return out

    @nb.njit(cache=True)
    def _tri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
        """2 * atan2 of the van Oosterom-Strackee formula, as tpw._solid_angle_tri."""
        a1x, a1y, a1z = ax - px, ay - py, az - pz
        a2x, a2y, a2z = bx - px, by - py, bz - pz
        a3x, a3y, a3z = cx - px, cy - py, cz - pz
        n1 = np.sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
        n2 = np.sqrt(a2x * a2x + a2y * a2y + a2z * a2z)
        n3 = np.sqrt(a3x * a3x + a3y * a3y + a3z * a3z)
        # summed from +0.0 like numpy's einsum, so that an in-plane point gets the same signed zero, hence the
        # same principal value arctan2(+-0, den < 0) = +-pi as the vendored kernel
        num = 0.0 + a1x * (a2y * a3z - a2z * a3y) + a1y * (a2z * a3x - a2x * a3z) + a1z * (a2x * a3y - a2y * a3x)
        den = (n1 * n2 * n3 + (a1x * a2x + a1y * a2y + a1z * a2z) * n3
               + (a2x * a3x + a2y * a3y + a2z * a3z) * n1
               + (a3x * a1x + a3y * a1y + a3z * a1z) * n2)
        return 2.0 * np.arctan2(num, den)

    @nb.njit(parallel=True, cache=True)
    def _solid_angle(pts, corners):
        """-1/2 * sum over the four diagonal triangles (0,1,2), (0,2,3), (1,2,3), (1,3,0): tpw.solid_angle."""
        m, k = pts.shape[0], corners.shape[0]
        out = np.empty((m, k))
        for i in nb.prange(m):
            px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
            for q in range(k):
                c = corners[q]
                s = _tri(px, py, pz, c[0, 0], c[0, 1], c[0, 2], c[1, 0], c[1, 1], c[1, 2], c[2, 0], c[2, 1], c[2, 2])
                s += _tri(px, py, pz, c[0, 0], c[0, 1], c[0, 2], c[2, 0], c[2, 1], c[2, 2], c[3, 0], c[3, 1], c[3, 2])
                s += _tri(px, py, pz, c[1, 0], c[1, 1], c[1, 2], c[2, 0], c[2, 1], c[2, 2], c[3, 0], c[3, 1], c[3, 2])
                s += _tri(px, py, pz, c[1, 0], c[1, 1], c[1, 2], c[3, 0], c[3, 1], c[3, 2], c[0, 0], c[0, 1], c[0, 2])
                out[i, q] = -0.5 * s
        return out


    @nb.njit(parallel=True, cache=True)
    def _source_vel(pts, sigma, centre, area, diag, origin, n_hat, t1, t2, xi, eta, live, tri1, tri2, far_diag):
        """Hess flat-panel source velocity, point-source form beyond far_diag diagonals: tpw.source_velocity."""
        m, k = pts.shape[0], centre.shape[0]
        out = np.zeros((m, 3))
        inv4pi = 1.0 / (4.0 * np.pi)
        for i in nb.prange(m):
            px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
            vx = vy = vz = 0.0
            for q in range(k):
                rx, ry, rz = px - centre[q, 0], py - centre[q, 1], pz - centre[q, 2]
                r2 = rx * rx + ry * ry + rz * rz
                lim = far_diag * diag[q]
                if r2 > lim * lim:
                    w = sigma[q] * area[q] * inv4pi / max(r2, 1e-30) ** 1.5
                    vx += rx * w; vy += ry * w; vz += rz * w
                    continue
                ox, oy, oz = px - origin[q, 0], py - origin[q, 1], pz - origin[q, 2]
                x = ox * t1[q, 0] + oy * t1[q, 1] + oz * t1[q, 2]
                y = ox * t2[q, 0] + oy * t2[q, 1] + oz * t2[q, 2]
                z = ox * n_hat[q, 0] + oy * n_hat[q, 1] + oz * n_hat[q, 2]
                u1 = u2 = 0.0
                for e in range(4):
                    e2 = (e + 1) % 4
                    dxi = xi[q, e2] - xi[q, e]; deta = eta[q, e2] - eta[q, e]
                    d = np.hypot(dxi, deta)
                    big_l = 0.0
                    if live[q, e]:
                        r1 = np.sqrt((x - xi[q, e]) ** 2 + (y - eta[q, e]) ** 2 + z * z)
                        rr2 = np.sqrt((x - xi[q, e2]) ** 2 + (y - eta[q, e2]) ** 2 + z * z)
                        num = r1 + rr2 + d
                        den = max(r1 + rr2 - d, 1e-10 * d + 1e-30)
                        big_l = np.log(max(num, 1e-30) / den)
                    d_safe = max(d, 1e-30)
                    u1 += deta / d_safe * big_l
                    u2 += -dxi / d_safe * big_l
                a, b = tri1[q], tri2[q]
                om = -(_tri(px, py, pz, a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2], a[2, 0], a[2, 1], a[2, 2])
                       + _tri(px, py, pz, b[0, 0], b[0, 1], b[0, 2], b[1, 0], b[1, 1], b[1, 2], b[2, 0], b[2, 1], b[2, 2]))
                s = sigma[q] * inv4pi
                vx += s * (u1 * t1[q, 0] + u2 * t2[q, 0] + om * n_hat[q, 0])
                vy += s * (u1 * t1[q, 1] + u2 * t2[q, 1] + om * n_hat[q, 1])
                vz += s * (u1 * t1[q, 2] + u2 * t2[q, 2] + om * n_hat[q, 2])
            out[i, 0] = vx; out[i, 1] = vy; out[i, 2] = vz
        return out


# -------------------------------------------------------------- 3 drop-ins

def source_velocity(pts, corners, sigma, chunk=300, far_diag=5.0):
    """tpw.source_velocity, compiled. The panel frames are the vendored ones (tpw._corner_frames), computed once."""
    if not ENABLED:
        return tpw.source_velocity(pts, corners, sigma, chunk, far_diag)
    corners = np.asarray(corners, dtype=np.float64)
    origin, n_hat, t1, t2, xi, eta = tpw._corner_frames(corners)
    proj = tpw._projected_corners(corners)
    centre = corners.mean(axis=1)
    d1 = corners[:, 2, :] - corners[:, 0, :]
    d2 = corners[:, 3, :] - corners[:, 1, :]
    area = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)
    diag = np.maximum(np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1))
    nxt = (1, 2, 3, 0)
    d = np.hypot(xi[:, nxt] - xi, eta[:, nxt] - eta)
    live = d > 1e-12 * np.maximum(d.max(axis=1, keepdims=True), 1e-30)
    c = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    return _source_vel(c(pts), c(sigma), c(centre), c(area), c(diag), c(origin), c(n_hat), c(t1), c(t2), c(xi), c(eta),
                       np.ascontiguousarray(live), c(proj[:, (0, 1, 2), :]), c(proj[:, (0, 2, 3), :]), float(far_diag))



def induced_velocity(points, p1, p2, strengths, core, chunk=300):
    """fk.induced_velocity, compiled. (M,3) out."""
    if not ENABLED:
        return fk.induced_velocity(points, p1, p2, strengths, core, chunk)
    c = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    return _induced(c(points), c(p1), c(p2), c(strengths), c(core))


def doublet_potential_matrix(pts, corners, far_diag=None, chunk=300):
    """tpw.doublet_potential_matrix, compiled (exact form; the far-field option falls back to the vendored one)."""
    if not ENABLED or far_diag is not None:
        return tpw.doublet_potential_matrix(pts, corners, far_diag, chunk)
    if len(corners) == 0:
        return np.zeros((len(pts), 0))
    return _solid_angle(np.ascontiguousarray(pts, dtype=np.float64),
                        np.ascontiguousarray(corners, dtype=np.float64)) / (4.0 * np.pi)


def wake_influence(colloc, wake, onset):
    """tpw.wake_influence with the compiled doublet potential; same strip and row bookkeeping."""
    if not ENABLED:
        return tpw.wake_influence(colloc, wake, onset)
    nrows = wake["nodes"].shape[0] - 1
    nspan = wake["mu"].shape[1]
    if nrows == 0:
        return np.zeros((len(colloc), nspan)), np.zeros(len(colloc))
    quads = tpw.wake_quads(wake, onset)
    per_strip = nrows + (1 if wake["closure"] else 0)
    phi_q = doublet_potential_matrix(colloc, quads)
    n_b = wake["n_bound"]
    p_bound = np.zeros((len(colloc), nspan))
    phi_known = np.zeros(len(colloc))
    for j in range(nspan):
        base = j * per_strip
        cols = phi_q[:, base:base + per_strip]
        if n_b == nrows:
            p_bound[:, j] = cols.sum(axis=1)
        else:
            p_bound[:, j] = cols[:, :n_b].sum(axis=1)
            mu_tail = wake["mu"][n_b:, j]
            phi_known += cols[:, n_b:nrows] @ mu_tail
            if wake["closure"]:
                phi_known += cols[:, -1] * wake["mu"][-1, j]
    return p_bound, phi_known


# ------------------------------------------------------------------ 4 main

def main():
    import time
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(2000, 3)); p1 = rng.normal(size=(8000, 3)); p2 = p1 + rng.normal(size=(8000, 3)) * 0.1
    g = rng.normal(size=8000); core = np.full(8000, 0.05)
    induced_velocity(pts[:5], p1[:5], p2[:5], g[:5], core[:5])                 # compile
    t = time.perf_counter(); a = induced_velocity(pts, p1, p2, g, core); ta = time.perf_counter() - t
    t = time.perf_counter(); b = fk.induced_velocity(pts, p1, p2, g, core); tb = time.perf_counter() - t
    print(f"induced_velocity: max rel diff {np.abs(a - b).max() / np.abs(b).max():.1e}, {tb:.2f} s -> {ta:.3f} s")
    q = rng.normal(size=(2000, 4, 3)) * 0.2 + rng.normal(size=(2000, 1, 3))
    doublet_potential_matrix(pts[:5], q[:5])
    t = time.perf_counter(); a = doublet_potential_matrix(pts, q); ta = time.perf_counter() - t
    t = time.perf_counter(); b = tpw.doublet_potential_matrix(pts, q); tb = time.perf_counter() - t
    print(f"doublet_potential_matrix: max abs diff {np.abs(a - b).max():.1e}, {tb:.2f} s -> {ta:.3f} s")
    import vent_mesh as vm
    pan = tpw.build_panels(vm.strut_mesh(n_c=24, nspan_half=16, h=1.0, alpha_deg=10.0, image=True, tip_taper=0.1))
    sig = rng.normal(size=len(pan["areas"])); probe = pan["centroids"][::3] + 0.05 * rng.normal(size=(len(pan["centroids"][::3]), 3))
    probe = np.vstack([probe, pan["points"][:, 0]])                          # on the tip line, as the wake row 0
    source_velocity(probe[:5], pan["corners"], sig)
    t = time.perf_counter(); a = source_velocity(probe, pan["corners"], sig); ta = time.perf_counter() - t
    t = time.perf_counter(); b = tpw.source_velocity(probe, pan["corners"], sig); tb = time.perf_counter() - t
    print(f"source_velocity: max rel diff {np.abs(a - b).max() / np.abs(b).max():.1e}, {tb:.2f} s -> {ta:.3f} s")


if __name__ == "__main__":
    main()
