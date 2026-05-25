from veros.core.operators import numpy as npx

from veros import veros_routine, veros_kernel, KernelOutput
from veros.variables import allocate, Variable
from veros.core import utilities
from veros.core.operators import update, update_add, at
from veros.core.thermodynamics import advect_tracer


def register_bgc_tracer(state, name, units="", long_name=""):
    """Register a BGC tracer and its transport variables in one call.

    Adds the tracer name to bgc_tracer_names and registers the three required
    variables: the tracer array, its AB tendency, and its surface forcing.
    Any additional setup-specific variables (e.g. relaxation targets) should
    be added to var_meta manually.
    """
    state.settings.bgc_tracer_names = state.settings.bgc_tracer_names + [name]
    state.var_meta.update(**{
        name: Variable(
            name, ("xt", "yt", "zt", "timesteps"), units, long_name
        ),
        f"d_{name}": Variable(
            f"d_{name}", ("xt", "yt", "zt", "timesteps"), f"{units}/s", f"{long_name} tendency"
        ),
        f"forc_{name}_surface": Variable(
            f"forc_{name}_surface", ("xt", "yt"), f"{units}/s", f"{long_name} surface forcing"
        ),
    })


@veros_kernel(static_args=("name",))
def _advect_bgc_tracer(state, name):
    vs = state.variables
    dtr = advect_tracer(state, getattr(vs, name)[..., vs.tau])
    new_dtr = update(getattr(vs, f"d_{name}"), at[..., vs.tau], dtr)
    return KernelOutput(**{f"d_{name}": new_dtr})


@veros_kernel(static_args=("name",))
def _ab_step_bgc_tracer(state, name):
    vs = state.variables
    settings = state.settings
    tr = getattr(vs, name)
    dtr = getattr(vs, f"d_{name}")
    new_tr = update(
        tr,
        at[:, :, :, vs.taup1],
        tr[:, :, :, vs.tau]
        + settings.dt_tracer
        * (
            (1.5 + settings.AB_eps) * dtr[:, :, :, vs.tau]
            - (0.5 + settings.AB_eps) * dtr[:, :, :, vs.taum1]
        )
        * vs.maskT,
    )
    return KernelOutput(**{name: new_tr})


@veros_kernel(static_args=("name",))
def _vertmix_bgc_tracer(state, name):
    vs = state.variables
    settings = state.settings
    tr = getattr(vs, name)
    forc = getattr(vs, f"forc_{name}_surface")

    a_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    b_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    c_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    d_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    delta = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]

    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    delta = update(
        delta, at[:, :, :-1], settings.dt_tracer / vs.dzw[npx.newaxis, npx.newaxis, :-1] * vs.kappaH[2:-2, 2:-2, :-1]
    )
    delta = update(delta, at[:, :, -1], 0.0)
    a_tri = update(a_tri, at[:, :, 1:], -delta[:, :, :-1] / vs.dzt[npx.newaxis, npx.newaxis, 1:])
    b_tri = update(b_tri, at[:, :, 1:], 1 + (delta[:, :, 1:] + delta[:, :, :-1]) / vs.dzt[npx.newaxis, npx.newaxis, 1:])
    b_tri_edge = 1 + delta / vs.dzt[npx.newaxis, npx.newaxis, :]
    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzt[npx.newaxis, npx.newaxis, :-1])

    d_tri = tr[2:-2, 2:-2, :, vs.taup1]
    d_tri = update_add(d_tri, at[:, :, -1], settings.dt_tracer * forc[2:-2, 2:-2] / vs.dzt[-1])

    sol = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    new_tr = update(tr, at[2:-2, 2:-2, :, vs.taup1], npx.where(water_mask, sol, tr[2:-2, 2:-2, :, vs.taup1]))

    new_tr = update(
        new_tr, at[..., vs.taup1], utilities.enforce_boundaries(new_tr[..., vs.taup1], settings.enable_cyclic_x)
    )

    return KernelOutput(**{name: new_tr})


@veros_routine
def integrate_bgc_tracers(state):
    vs = state.variables
    for name in state.settings.bgc_tracer_names:
        vs.update(_advect_bgc_tracer(state, name))
        vs.update(_ab_step_bgc_tracer(state, name))
        vs.update(_vertmix_bgc_tracer(state, name))
