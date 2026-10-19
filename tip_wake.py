"""Tip shedding: a doublet sheet shed from the pinched side edge of the wing.

The vendored solver sheds a wake from the trailing edge only. At a pinched tip the
loading then has to fall to zero inside the tip strip and leave the wing through
the trailing-edge corner, as one concentrated filament. The vortex-lattice codes
this repository is compared with (vlm-ventilation, vlm-wakeRel_cavityFlow, with
shedding 'right') shed rings from the tip chord as well - side-edge separation -
so the loading of the tip strip leaves along the whole chord. This module gives
the panel method the same option, as data that the ventilated solve folds in
exactly like the trailing-edge wake; the vendored code is not touched.

Five sections, and the same three discipline rules: nothing imports downward,
every block is a pure function over explicit arguments with all mutable data in
an explicit state dictionary, and prompts, prints and figures appear only under
main().

    1 imports       numpy; thick_panel_wing
    2 the edge      the tip line, the panel pairs across it, and a sharp tip
    3 the sheets    one vendored-format wake per tip
    4 the fold      extra columns for assemble_system, the strength sync, and a
                    steady solve for a wing without free surface
    5 main          a smoke test

Three decisions are worth stating before the code.

**The sheet is a vendored wake dictionary.** Its "span" runs along the tip chord,
from the leading edge to the trailing edge, and its rows run downstream; each
chordwise strip m carries the doublet jump mu[k_up[m]] - mu[k_low[m]] of the
upper and lower tip-strip panels at that station, the panel-method form of the
VLM rule that a tip ring is shed with the circulation of the tip panel it leaves.
`tpw.wake_influence` and `tpw.assemble_system` then take it as they take the
trailing-edge wake: one more influence column per strip, folded onto two panels.

**The orientation is fixed by the trailing-edge corner, not chosen.** The sheet
of a tip and the trailing-edge wake share the filament that leaves the corner.
With the strip order below, the sheet's last strip carries exactly the strength
of the trailing-edge wake's tip strip (same two panels), and its corner filament
is the opposite of the wake's: the concentrated tip vortex is removed and its
circulation leaves through the side edge instead, strip by strip. The span-node 0
tip is ordered leading edge to trailing edge, the span-node nspan tip the reverse.

**Steady and frozen only.** The sheets are straight along the onset and are not
relaxed, shed or projected: the ventilated comparisons use a frozen wake, and an
image tip sheet would need the projection `vent_mesh.symmetrise_wake` gives the
trailing-edge wake before it could be convected.
"""

# --------------------------------------------------------------- 1 imports

import numpy as np

import thick_panel_wing as tpw


# --------------------------------------------------------------- 2 the edge

def tip_edge(points, pan, side):
    """Tip line nodes and the panel pairs across it. -> (nodes (n_c+1, 3), k_up, k_low)

    side = 0 for span node 0, -1 for span node nspan. The wrap runs from the
    lower trailing edge (i = 0) round the leading edge (i = n_c) to the upper
    trailing edge (i = nwrap); at a pinched station both surfaces share the
    camber-line nodes, so the upper nodes n_c..nwrap are the tip line. Chordwise
    interval m (from the leading edge) has upper panel n_c + m and lower panel
    n_c - 1 - m in the tip strip.
    """
    nwrap, nspan = pan["nwrap"], pan["nspan"]
    if nwrap % 2:
        raise ValueError(f"tip shedding needs an even wrap count; got {nwrap}")
    n_c = nwrap // 2
    j_node = 0 if side == 0 else nspan
    j_strip = 0 if side == 0 else nspan - 1
    m = np.arange(n_c)
    k_up = (n_c + m) * nspan + j_strip
    k_low = (n_c - 1 - m) * nspan + j_strip
    nodes = np.asarray(points, float)[n_c:, j_node, :].copy()      # LE -> TE
    pinch = float(np.abs(points[n_c:, j_node] - points[n_c::-1, j_node]).max())
    if pinch > 1e-9 * max(float(np.abs(points).max()), 1.0):
        raise ValueError(f"span node {j_node} is not a pinched tip "
                         f"(surfaces {pinch:.2e} apart); nothing to shed from")
    if side != 0:                                                   # TE -> LE
        nodes, k_up, k_low = nodes[::-1], k_up[::-1], k_low[::-1]
    return nodes, k_up, k_low


def taper_tips(points, length, ends=(0, -1)):
    """Thin the section linearly to zero over `length` from each pinched end. -> points

    A pinched tip closes the section within one strip, a wedge that opens to a
    blunt edge as the strips narrow, and a blunt edge supports no finite jump: a
    tip sheet shed from it fades under refinement. Thinning over a fixed length
    keeps the tip a wedge of fixed angle. The span runs along y and the thickness
    along z, i.e. before any pitch; the camber line (mean of the paired wrap
    nodes) is kept. ends: 0 for span node 0, -1 for span node nspan.
    """
    points = np.array(points, dtype=float)
    if length <= 0.0:
        return points
    nspan = points.shape[1] - 1
    mid = 0.5 * (points + points[::-1])                    # camber, per wrap pair
    for end in ends:
        y0 = points[0, 0 if end == 0 else nspan, 1]
        for j in range(nspan + 1):
            f = abs(points[0, j, 1] - y0) / length
            if 0.0 < f < 1.0:
                points[:, j, 2] = mid[:, j, 2] + f * (points[:, j, 2] - mid[:, j, 2])
    return points


# ------------------------------------------------------------- 3 the sheets

def add_tip_wakes(fluid, sides=(0, -1)):
    """Attach one tip sheet per pinched tip to the fluid state. -> fluid

    The sheets take the discretisation of the trailing-edge wake (row lengths,
    closure, bound rows, filament core), so the two meet at the corner row for
    row. sides: which tips shed; (0, -1) for both, which on the doubled strut of
    vent_mesh is the free tip and its image.
    """
    te = fluid["wake"]
    if te is None:
        raise ValueError("tip shedding needs a lifting state with a wake")
    pan = fluid["panels"]
    out = []
    for side in sides:
        nodes, k_up, k_low = tip_edge(pan["points"], pan, side)
        wake = tpw.init_wake(nodes, fluid["onset"], te["steps"],
                             n_bound=te["n_bound"], closure=te["closure"],
                             core=te["core"])
        out.append({"wake": wake, "k_up": k_up, "k_low": k_low, "side": side})
    fluid["tip_wakes"] = out
    return fluid


# --------------------------------------------------------------- 4 the fold

def fold_arrays(fluid, colloc, p_bound, phi_known, k_up, k_low):
    """Append the tip strips to the trailing-edge fold. -> (p_bound, phi_known, k_up, k_low)

    The arrays are what tpw.assemble_system takes; with no tip sheets they are
    returned unchanged, so the trailing-edge-only system is reproduced bit for bit.
    """
    tips = fluid.get("tip_wakes") or []
    if not tips:
        return p_bound, phi_known, k_up, k_low
    cols, kus, kls = [p_bound], [np.asarray(k_up)], [np.asarray(k_low)]
    phi = np.zeros(len(colloc)) if phi_known is None else np.array(phi_known, float)
    for t in tips:
        pb, pk = tpw.wake_influence(colloc, t["wake"], fluid["onset"])
        cols.append(pb); kus.append(t["k_up"]); kls.append(t["k_low"])
        phi = phi + pk
    return np.hstack(cols), phi, np.concatenate(kus), np.concatenate(kls)


def sync_strengths(fluid, mu):
    """Slave the bound rows of every tip sheet to the current jumps (in place)."""
    for t in fluid.get("tip_wakes") or []:
        tpw.set_bound_strengths(t["wake"], mu, t["k_up"], t["k_low"])


def solve_steady(state):
    """One steady solve of a wing with no free surface, tip sheets folded in. -> state

    The vendored solve_once with the fold extended; frozen wake geometry (the tip
    sheets are not relaxed). tpw.get_loads then reads the strengths as usual.
    """
    pan = state["panels"]
    if state["blocks"] is None or state["blocks_version"] != state["geom_version"]:
        state["blocks"] = tpw.assemble_body(pan)
        state["blocks_version"] = state["geom_version"]
    sigma = tpw.source_strengths(pan["normals"], state["u_rel"])
    pb, pk = tpw.wake_influence(pan["centroids"], state["wake"], state["onset"])
    pb, pk, k_up, k_low = fold_arrays(state, pan["centroids"], pb, pk, pan["k_up"], pan["k_low"])
    n = len(sigma)
    a_mat, rhs = tpw.assemble_system(state["blocks"]["D"], state["blocks"]["S"], pb, pk, k_up, k_low,
                                     np.zeros(n, dtype=np.int8), np.zeros(n), sigma)
    mu = np.linalg.solve(a_mat, rhs)
    state["mu"], state["sigma"] = mu, sigma
    tpw.set_bound_strengths(state["wake"], mu, pan["k_up"], pan["k_low"])
    sync_strengths(state, mu)
    return state


# ------------------------------------------------------------------ 5 main

def main():
    import freewake_kernels as fk
    import vent_mesh as vm
    points = vm.strut_mesh(n_c=8, nspan_half=8, h=1.0, chord=1.0, alpha_deg=10.0)
    onset = fk.make_onset(np.array([-10.0, 10.0]), np.array([1.0, 1.0]))
    fluid = tpw.init_state(points, onset, 1.0, config={"lifting": True, "nwake": 6})
    add_tip_wakes(fluid)
    for t in fluid["tip_wakes"]:
        print(f"tip {t['side']:+d}: {len(t['k_up'])} strips, nodes "
              f"{t['wake']['nodes'].shape}, first node {t['wake']['nodes'][0, 0]}")


if __name__ == "__main__":
    main()
