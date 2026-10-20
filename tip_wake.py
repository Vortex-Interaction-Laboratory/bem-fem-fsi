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
    5 relaxation    the node velocities of every sheet, one streamline remarch
                    of all of them, and the relaxed steady solve of a wing
    6 main          a smoke test

Four decisions are worth stating before the code.

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

**Relaxed together with the trailing-edge wake, or not at all.** The sheets start
straight along the onset. Relaxation remarches every sheet along the velocity of
the whole field - body, sources, trailing-edge wake and every tip sheet - and
then pins the corner filament of each tip sheet back onto the trailing-edge
wake's, so the cancellation at the corner survives the convection. The first
segment of every filament, trailing-edge or tip, follows the velocity just behind
its edge rather than the singular one on it (see relax_step). The body ring
edges lying on a shedding tip line are left out of that velocity, as the vendored
code leaves out the trailing-edge ones: they cancel the sheet's first row, which
is never emitted. A free-surface image sheet is projected, not convected, by
`vent_solve`. The sheets are not shed in a march.
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


def solve_steady(state, wake="frozen"):
    """Steady solve of a wing with no free surface, tip sheets folded in. -> state

    wake='frozen' is the vendored solve_once with the fold extended, on the
    current wake geometry; 'relax' iterates it with section 5. tpw.get_loads then
    reads the strengths as usual.
    """
    if wake == "relax":
        return relax_steady(state)
    if wake != "frozen":
        raise ValueError(f"unknown wake mode '{wake}'; choose 'frozen' or 'relax'")
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


# ----------------------------------------------------------- 5 relaxation

def sheets(fluid):
    """The trailing-edge wake followed by the tip sheets. -> list of wake dicts"""
    return [fluid["wake"]] + [t["wake"] for t in fluid.get("tip_wakes") or []]


def body_segments_open(fluid):
    """Body ring segments without the edges lying on a shedding tip line.

    tpw.body_segments already drops the trailing-edge edges, which cancel the
    trailing-edge wake's first row; here the tip-line edges go the same way, for
    the same reason. With no tip sheets this is tpw.body_segments unchanged.
    """
    p1, p2, g, core = tpw.body_segments(fluid["panels"], fluid["mu"])
    tips = fluid.get("tip_wakes") or []
    if not tips:
        return p1, p2, g, core
    scale = 1e-9 * max(float(np.abs(fluid["panels"]["points"]).max()), 1.0)
    on = np.zeros(len(p1), dtype=bool)
    for t in tips:
        line = t["wake"]["nodes"][0]                        # the tip line, row 0
        d1 = np.linalg.norm(p1[:, None, :] - line[None], axis=2).min(axis=1)
        d2 = np.linalg.norm(p2[:, None, :] - line[None], axis=2).min(axis=1)
        on |= (d1 < scale) & (d2 < scale)
    keep = ~on
    return p1[keep], p2[keep], g[keep], core[keep]


def node_velocities(fluid):
    """Advection velocity at the nodes of every sheet. -> list, one (n, 3) per sheet

    tpw.wake_node_velocity with the tip sheets among both the targets and the
    sources: onset + sources + body rings + every sheet's segments.
    """
    pan, ws = fluid["panels"], sheets(fluid)
    pts = np.concatenate([w["nodes"].reshape(-1, 3) for w in ws])
    v = fluid["onset"](pts) + tpw.source_velocity(pts, pan["corners"], fluid["sigma"])
    segs = [body_segments_open(fluid)] + [tpw.wake_segments(w, fluid["onset"]) for w in ws]
    p1, p2, g, core = (np.concatenate([s[i] for s in segs]) for i in range(4))
    v = v + tpw.induced_velocity(pts, p1, p2, g, core)
    sizes = np.cumsum([w["nodes"].reshape(-1, 3).shape[0] for w in ws])[:-1]
    return np.split(v, sizes)


def set_core(fluid, core):
    """Give every sheet the filament core `core` (in place). -> fluid

    The vendored core is RC_WAKE_FRACTION times the spanwise spacing: 0.006 c on
    a strut of AR_h 0.5 with 16 strips, and it only enters the convection. With a
    tip sheet that rolls up, filaments then pass within a core of one another and
    the relaxed lift explodes (CL 1.65 against 0.20 frozen at 12.5 deg); a core
    on the chord, as the vortex-lattice codes use (0.05 c), keeps it bounded.
    """
    for w in sheets(fluid):
        w["core"] = float(core)
    return fluid


def pin_corners(fluid):
    """Put each tip sheet's corner filament back on the trailing-edge wake's (in place)."""
    te = fluid["wake"]["nodes"]
    for t in fluid.get("tip_wakes") or []:
        nodes = t["wake"]["nodes"]
        if t["side"] == 0:                     # leading edge -> trailing edge
            nodes[:, -1] = te[:, 0]
        else:                                  # trailing edge -> leading edge
            nodes[:, 0] = te[:, -1]


def relax_step(fluid, v_list, omega):
    """One under-relaxed streamline remarch of every sheet. -> max near-field move

    v_list is node_velocities(fluid); row 0 of each sheet stays pinned on its
    edge, as in tpw.relax_wake_step, and the corner filaments are re-joined.

    The first segment of each filament leaves its edge along the velocity at
    row 1, not at row 0. Row 0 lies ON the body edge, where the discrete velocity
    is the singular surface limit of the rings and sources rather than the flow
    leaving the edge: on a straight wing at 2 deg it points almost normal to the
    onset, (-0.02, 0, 0.07), and on a 45 deg swept wing it has a spanwise
    component of 0.68, which folds the wake and makes the vendored relaxation
    oscillate without converging (CL between 0.01 and 0.12 against 0.125 frozen).
    """
    delta = 0.0
    for w, v in zip(sheets(fluid), v_list):
        v = np.array(v, dtype=float).reshape(w["nodes"].shape)
        if len(v) > 1:
            v[0] = v[1]
        delta = max(delta, tpw.relax_wake_step(w, v.reshape(-1, 3), omega, fluid["x_near"]))
    pin_corners(fluid)
    return delta


def relax_steady(state, max_iter=None, tol=None, omega=None):
    """Steady solve with every sheet relaxed, wing without free surface. -> state

    The vendored relaxation loop (tpw.solve, wake='relax') with the tip sheets in
    the field and among the convected nodes, and the first segment of every
    filament along the velocity just behind its edge (relax_step). With or
    without tip sheets, so that the two are relaxed by the same rule.
    """
    max_iter = tpw.MAX_ITER if max_iter is None else max_iter
    tol = tpw.TOL if tol is None else tol
    omega = tpw.OMEGA if omega is None else omega
    if state["blocks"] is None or state["blocks_version"] != state["geom_version"]:
        state["blocks"] = tpw.assemble_body(state["panels"])
        state["blocks_version"] = state["geom_version"]
    for it in range(max_iter):
        solve_steady(state)
        delta = relax_step(state, node_velocities(state), omega)
        state["wake_version"] += 1
        state["residuals"].update(wake_disp_chords=delta / state["chord_mean"],
                                  iterations=it + 1)
        if delta < tol * state["chord_mean"]:
            break
    return solve_steady(state)


# ------------------------------------------------------------------ 6 main

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
