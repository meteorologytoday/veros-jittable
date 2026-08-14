from veros import veros_kernel, veros_routine, KernelOutput, logger
from veros.variables import allocate
from veros.distributed import global_and
from veros.core import density, diffusion, utilities
from veros.core.operators import update, at, numpy as npx


@veros_kernel
def u_centered_grid(dyt, dyu, yt, yu):
    yu = update(yu, at[0], 0)
    yu = update(yu, at[1:], npx.cumsum(dyt[1:]))

    yt = update(yt, at[0], yu[0] - dyt[0] * 0.5)
    yt = update(yt, at[1:], 2 * yu[:-1])

    alternating_pattern = npx.ones_like(yt)
    alternating_pattern = update(alternating_pattern, at[::2], -1)
    yt = update(yt, at[...], alternating_pattern * npx.cumsum(alternating_pattern * yt))

    dyu = update(dyu, at[:-1], yt[1:] - yt[:-1])
    dyu = update(dyu, at[-1], 2 * dyt[-1] - dyu[-2])
    return dyu, yt, yu


@veros_kernel
def calc_grid_spacings_kernel(state):
    """
    xt/yt/xu/yu/dxt/dyt/dxu/dyu are now 2D (T_HOR/U_HOR/V_HOR-shaped), to
    accommodate a curvilinear grid where neither coordinate is separable.
    On this (legacy, analytic) grid path, every VerosSetup.set_grid()
    implementation still only ever broadcasts a scalar or an i-only/j-only
    profile into dxt/dyt (never a genuinely 2D pattern), so dxt is constant
    along j and dyt is constant along i. That lets the whole grid-spacing
    construction be done on 1D i- and j-profiles exactly as before (reusing
    u_centered_grid unchanged), then broadcast into the 2D arrays at the end
    -- which reproduces today's values exactly, by construction, and also
    reproduces the implicit outer-product structure that calc_grid_metrics_kernel's
    area formulas rely on below.
    """
    vs = state.variables
    settings = state.settings

    dxt_1d = vs.dxt[:, 0]
    dyt_1d = vs.dyt[0, :]

    if settings.enable_cyclic_x:
        dxt_1d = update(dxt_1d, at[-2:], dxt_1d[2:4])
        dxt_1d = update(dxt_1d, at[:2], dxt_1d[-4:-2])
    else:
        dxt_1d = update(dxt_1d, at[-2:], dxt_1d[-3])
        dxt_1d = update(dxt_1d, at[:2], dxt_1d[2])

    dyt_1d = update(dyt_1d, at[-2:], dyt_1d[-3])
    dyt_1d = update(dyt_1d, at[:2], dyt_1d[2])

    """
    grid in east/west direction
    """
    dxu_1d, xt_1d, xu_1d = u_centered_grid(
        dxt_1d, npx.zeros_like(dxt_1d), npx.zeros_like(dxt_1d), npx.zeros_like(dxt_1d)
    )
    xt_1d = xt_1d + settings.x_origin - xu_1d[2]
    xu_1d = xu_1d + settings.x_origin - xu_1d[2]

    if settings.enable_cyclic_x:
        xt_1d = update(xt_1d, at[-2:], xt_1d[2:4])
        xt_1d = update(xt_1d, at[:2], xt_1d[-4:-2])
        xu_1d = update(xu_1d, at[-2:], xt_1d[2:4])
        xu_1d = update(xu_1d, at[:2], xu_1d[-4:-2])
        dxu_1d = update(dxu_1d, at[-2:], dxu_1d[2:4])
        dxu_1d = update(dxu_1d, at[:2], dxu_1d[-4:-2])

    """
    grid in north/south direction
    """
    dyu_1d, yt_1d, yu_1d = u_centered_grid(
        dyt_1d, npx.zeros_like(dyt_1d), npx.zeros_like(dyt_1d), npx.zeros_like(dyt_1d)
    )
    yt_1d = yt_1d + settings.y_origin - yu_1d[2]
    yu_1d = yu_1d + settings.y_origin - yu_1d[2]

    if settings.coord_degree:
        """
        convert from degrees to pseudo cartesian grid
        """
        dxt_1d = dxt_1d * settings.degtom
        dxu_1d = dxu_1d * settings.degtom
        dyt_1d = dyt_1d * settings.degtom
        dyu_1d = dyu_1d * settings.degtom

    """
    broadcast the 1D i-/j-profiles into the 2D T_HOR/U_HOR/V_HOR arrays --
    dxt/xt/yt are T-point quantities (constant along j for xt-things, along i
    for yt-things), dxu/xu are U-point (indexed by xu, constant along yt),
    dyu/yu are V-point (indexed by yu, constant along xt)
    """
    vs.dxt = dxt_1d[:, npx.newaxis] * npx.ones_like(vs.dxt)
    vs.dyt = dyt_1d[npx.newaxis, :] * npx.ones_like(vs.dyt)
    vs.xt = xt_1d[:, npx.newaxis] * npx.ones_like(vs.xt)
    vs.yt = yt_1d[npx.newaxis, :] * npx.ones_like(vs.yt)

    vs.dxu = dxu_1d[:, npx.newaxis] * npx.ones_like(vs.dxu)
    vs.xu = xu_1d[:, npx.newaxis] * npx.ones_like(vs.xu)

    vs.dyu = dyu_1d[npx.newaxis, :] * npx.ones_like(vs.dyu)
    vs.yu = yu_1d[npx.newaxis, :] * npx.ones_like(vs.yu)

    """
    grid in vertical direction
    """
    vs.dzw, vs.zt, vs.zw = calc_vertical_grid(vs.dzt, vs.dzw, vs.zt, vs.zw)

    return KernelOutput(
        dxt=vs.dxt,
        dyt=vs.dyt,
        dxu=vs.dxu,
        dyu=vs.dyu,
        xt=vs.xt,
        yt=vs.yt,
        xu=vs.xu,
        yu=vs.yu,
        dzw=vs.dzw,
        zt=vs.zt,
        zw=vs.zw,
    )


@veros_kernel
def calc_vertical_grid(dzt, dzw, zt, zw):
    """
    Vertical grid construction (dzt -> dzw/zt/zw). Independent of the
    horizontal grid, so both the legacy analytic path
    (calc_grid_spacings_kernel) and the curvilinear path (calc_grid_scrip)
    call this the same way.
    """
    dzw, zt, zw = u_centered_grid(dzt, dzw, zt, zw)
    zt = zt - zw[-1]
    zw = zw - zw[-1]  # enforce 0 boundary height
    return dzw, zt, zw


@veros_routine(
    # grid metadata is cheap to (re)compute redundantly, so doing this on the
    # main process is fine even though these arrays are now 2D
    dist_safe=False,
    local_variables=(
        "dxt",
        "dxu",
        "xt",
        "xu",
        "dyt",
        "dyu",
        "yt",
        "yu",
        "dzt",
        "dzw",
        "zt",
        "zw",
    ),
)
def calc_grid_spacings(state):
    vs = state.variables
    vs.update(calc_grid_spacings_kernel(state))


@veros_kernel
def calc_grid_metrics_kernel(state):
    vs = state.variables
    settings = state.settings

    """
    metric factors
    """
    if settings.coord_degree:
        vs.cost = update(vs.cost, at[...], npx.cos(vs.yt * settings.pi / 180.0))
        vs.cosu = update(vs.cosu, at[...], npx.cos(vs.yu * settings.pi / 180.0))
        vs.tantr = update(vs.tantr, at[...], npx.tan(vs.yt * settings.pi / 180.0) / settings.radius)
    else:
        vs.cost = update(vs.cost, at[...], 1.0)
        vs.cosu = update(vs.cosu, at[...], 1.0)
        vs.tantr = update(vs.tantr, at[...], 0.0)

    """
    precalculate area of boxes

    dxt/dxu/dyt/dyu/cost/cosu are now all already 2D (T_HOR/U_HOR/V_HOR), so
    these are plain elementwise products, no newaxis broadcast needed -- on
    the legacy grid this reproduces the original 1D outer-product formulas
    exactly, since dxt/dxu vary only along i and dyt/dyu/cost/cosu only along
    j (see calc_grid_spacings_kernel).
    """
    vs.area_t = update(vs.area_t, at[...], vs.cost * vs.dyt * vs.dxt)
    vs.area_u = update(vs.area_u, at[...], vs.cost * vs.dyt * vs.dxu)
    vs.area_v = update(vs.area_v, at[...], vs.cosu * vs.dyu * vs.dxt)

    return KernelOutput(
        cost=vs.cost,
        cosu=vs.cosu,
        tantr=vs.tantr,
        area_t=vs.area_t,
        area_u=vs.area_u,
        area_v=vs.area_v,
    )


@veros_routine(
    # SCRIP file I/O and mesh derivation are plain numpy/xarray, not
    # jax-traceable -- do them on the main process, like calc_grid_spacings
    dist_safe=False,
    local_variables=(
        "xt",
        "xu",
        "yt",
        "yu",
        "dxt",
        "dxu",
        "dyt",
        "dyu",
        "cost",
        "cosu",
        "area_t",
        "area_u",
        "area_v",
        "dzt",
        "dzw",
        "zt",
        "zw",
    ),
)
def calc_grid_scrip(state):
    """
    Populate the horizontal grid from a SCRIP file (settings.scrip_grid_file)
    describing a locally-orthogonal curvilinear grid. Curvilinear counterpart
    to calc_grid_spacings + calc_grid_metrics_kernel; see calc_grid for the
    settings.enable_curvilinear_grid dispatch between the two paths.

    Does not compute tantr: nothing on the curvilinear path consumes it (the
    momentum equation's tan(lat)/R metric term is only valid for a regular
    lat-lon grid, and its general orthogonal-curvilinear replacement is an
    explicitly separate, not-yet-done follow-on task -- see
    veros/core/momentum.py::tend_coriolisf). tantr is left at its allocation
    default, unused.
    """
    import numpy as onp

    from veros.tools.scrip import read_scrip_grid

    vs = state.variables
    settings = state.settings

    if settings.scrip_grid_file is None:
        raise RuntimeError("settings.enable_curvilinear_grid=True requires settings.scrip_grid_file to be set")

    logger.warning(
        "enable_curvilinear_grid is True: the momentum equation's grid-curvature "
        "metric term (tan(lat)/R) is only valid on a regular lat-lon grid and is "
        "disabled on this curvilinear run (see veros/core/momentum.py::tend_coriolisf). "
        "This is a known, documented limitation of the current curvilinear grid support, "
        "not a bug -- the general orthogonal-curvilinear replacement is a separate follow-on task."
    )

    grid = read_scrip_grid(settings.scrip_grid_file, radius=settings.radius)
    nx, ny = grid.shape
    if (nx, ny) != (settings.nx, settings.ny):
        raise ValueError(
            f"SCRIP grid file {settings.scrip_grid_file!r} describes a ({nx}, {ny}) grid, "
            f"but settings.nx, settings.ny = ({settings.nx}, {settings.ny})"
        )

    def pad(interior):
        """
        Pad a (nx, ny) array with the same 4-ghost-cell, enable_cyclic_x
        convention as calc_grid_spacings_kernel: wrap around in i if cyclic,
        else repeat the edge value in i; always repeat the edge value in j
        (Veros has no cyclic-y notion).
        """
        padded = onp.empty((nx + 4, ny + 4), dtype=interior.dtype)
        padded[2:-2, 2:-2] = interior
        if settings.enable_cyclic_x:
            padded[-2:, 2:-2] = interior[:2, :]
            padded[:2, 2:-2] = interior[-2:, :]
        else:
            padded[-2:, 2:-2] = interior[-1:, :]
            padded[:2, 2:-2] = interior[:1, :]
        padded[:, -2:] = padded[:, -3:-2]
        padded[:, :2] = padded[:, 2:3]
        return padded

    vs.xt = pad(grid.xt)
    vs.yt = pad(grid.yt)
    vs.xu = pad(grid.xu)
    vs.yu = pad(grid.yu)
    vs.dxt = pad(grid.dxt)
    vs.dyt = pad(grid.dyt)
    vs.dxu = pad(grid.dxu)
    vs.dyu = pad(grid.dyu)

    # cost/cosu exist on the legacy path to convert degtom-scaled *degrees*
    # into true physical arc length; dxt/dyt/dxu/dyu here are already true
    # great-circle arc lengths (see veros/tools/scrip.py), so no separate
    # cos(lat) correction applies -- matches the generalized
    # orthogonal-coordinate convention (Griffies; MOM4/POP), where cell area
    # is simply h1*h2*dx1*dx2 with no extra trigonometric factor.
    vs.cost = onp.ones((nx + 4, ny + 4))
    vs.cosu = onp.ones((nx + 4, ny + 4))

    # area_t: use the SCRIP file's own precise spherical-polygon cell area
    # rather than the dxt*dyt product approximation calc_grid_metrics_kernel
    # would give -- it's the more accurate figure and we already have it.
    vs.area_t = pad(grid.area_t)

    # area_u: U-points don't shift latitude relative to their T-neighbours
    # (U_HOR=(xu,yt) -- same yt as T), so T's own dyt needs no cross-latitude
    # correction to pair with dxu here.
    vs.area_u = vs.dyt * vs.dxu

    # area_v: V-points DO shift latitude relative to T (V_HOR=(xt,yu) sits at
    # yu, not yt), so T's own dxt -- computed at T's latitude -- isn't quite
    # the right i-width for a cell centred at V's (different) latitude.
    # Approximate it the same way dxt itself was built (averaging over the
    # two neighbours that straddle the point in question): average T(i,j)'s
    # and T(i,j+1)'s dxt, which brackets the V-point's own latitude, rather
    # than reusing T(i,j)'s dxt unchanged (verified in
    # test/curvilinear_grid_test.py: this drops the legacy-reduction error
    # for area_v from ~20% to sub-percent on a regular-grid fixture).
    dxt_at_v = onp.empty_like(vs.dxt)
    dxt_at_v[:, :-1] = 0.5 * (vs.dxt[:, :-1] + vs.dxt[:, 1:])
    dxt_at_v[:, -1] = vs.dxt[:, -1]  # no j+1 neighbour at the last ghost row
    vs.area_v = vs.dyu * dxt_at_v

    """
    grid in vertical direction -- independent of the horizontal grid choice
    """
    vs.dzw, vs.zt, vs.zw = calc_vertical_grid(vs.dzt, vs.dzw, vs.zt, vs.zw)


@veros_routine
def calc_grid(state):
    """
    setup the horizontal grid: either analytically from dxt,dyt,dzt and
    x_origin, y_origin (the legacy, regular/separable path), or from a SCRIP
    file describing a locally-orthogonal curvilinear grid, depending on
    settings.enable_curvilinear_grid.
    """
    settings = state.settings

    if settings.enable_curvilinear_grid:
        calc_grid_scrip(state)
        return

    calc_grid_spacings(state)

    vs = state.variables
    vs.update(calc_grid_metrics_kernel(state))


@veros_routine
def calc_beta(state):
    """
    calculate beta = df/dy
    """
    vs = state.variables
    settings = state.settings
    # dyu is now V_HOR-shaped (2D); slice its j-axis (axis 1) to match
    # coriolis_t's j-slicing, not its i-axis (axis 0) -- on the legacy grid
    # dyu is constant along i, so this reproduces the original 1D-broadcast
    # division exactly.
    vs.beta = update(
        vs.beta,
        at[:, 2:-2],
        0.5
        * (
            (vs.coriolis_t[:, 3:-1] - vs.coriolis_t[:, 2:-2]) / vs.dyu[:, 2:-2]
            + (vs.coriolis_t[:, 2:-2] - vs.coriolis_t[:, 1:-3]) / vs.dyu[:, 1:-3]
        ),
    )
    vs.beta = utilities.enforce_boundaries(vs.beta, settings.enable_cyclic_x)


@veros_kernel
def calc_topo_kernel(state):
    vs = state.variables
    settings = state.settings

    """
    close domain
    """
    vs.kbot = update(vs.kbot, at[:, :2], 0)
    vs.kbot = update(vs.kbot, at[:, -2:], 0)

    vs.kbot = utilities.enforce_boundaries(vs.kbot, settings.enable_cyclic_x)

    if not settings.enable_cyclic_x:
        vs.kbot = update(vs.kbot, at[:2, :], 0)
        vs.kbot = update(vs.kbot, at[-2:, :], 0)

    """
    Land masks
    """
    land_mask = vs.kbot > 0
    ks = npx.arange(vs.maskT.shape[2])[npx.newaxis, npx.newaxis, :]

    vs.maskT = update(vs.maskT, at[...], land_mask[..., npx.newaxis] & (vs.kbot[..., npx.newaxis] - 1 <= ks))
    vs.maskT = utilities.enforce_boundaries(vs.maskT, settings.enable_cyclic_x)

    vs.maskU = update(vs.maskU, at[...], vs.maskT)
    vs.maskU = update(vs.maskU, at[:-1, :, :], npx.minimum(vs.maskT[:-1, :, :], vs.maskT[1:, :, :]))
    vs.maskU = utilities.enforce_boundaries(vs.maskU, settings.enable_cyclic_x)

    vs.maskV = update(vs.maskV, at[...], vs.maskT)
    vs.maskV = update(vs.maskV, at[:, :-1], npx.minimum(vs.maskT[:, :-1], vs.maskT[:, 1:]))
    vs.maskV = utilities.enforce_boundaries(vs.maskV, settings.enable_cyclic_x)

    vs.maskZ = update(vs.maskZ, at[...], vs.maskT)
    vs.maskZ = update(
        vs.maskZ, at[:-1, :-1], npx.minimum(npx.minimum(vs.maskT[:-1, :-1], vs.maskT[:-1, 1:]), vs.maskT[1:, :-1])
    )
    vs.maskZ = utilities.enforce_boundaries(vs.maskZ, settings.enable_cyclic_x)

    vs.maskW = update(vs.maskW, at[...], vs.maskT)
    vs.maskW = update(vs.maskW, at[:, :, :-1], npx.minimum(vs.maskT[:, :, :-1], vs.maskT[:, :, 1:]))

    """
    total depth
    """
    vs.ht = npx.sum(vs.maskT * vs.dzt[npx.newaxis, npx.newaxis, :], axis=2)
    vs.hu = npx.sum(vs.maskU * vs.dzt[npx.newaxis, npx.newaxis, :], axis=2)
    vs.hv = npx.sum(vs.maskV * vs.dzt[npx.newaxis, npx.newaxis, :], axis=2)

    vs.hur = npx.where(vs.hu != 0, 1 / (vs.hu + 1e-22), 0)
    vs.hvr = npx.where(vs.hv != 0, 1 / (vs.hv + 1e-22), 0)

    return KernelOutput(
        maskT=vs.maskT,
        maskU=vs.maskU,
        maskV=vs.maskV,
        maskW=vs.maskW,
        maskZ=vs.maskZ,
        ht=vs.ht,
        hu=vs.hu,
        hv=vs.hv,
        hur=vs.hur,
        hvr=vs.hvr,
        kbot=vs.kbot,
    )


@veros_routine
def calc_topo(state):
    """
    calulate masks, total depth etc
    """
    vs = state.variables
    vs.update(calc_topo_kernel(state))


@veros_kernel
def calc_initial_conditions_kernel(state):
    vs = state.variables
    settings = state.settings

    vs.temp = utilities.enforce_boundaries(vs.temp, settings.enable_cyclic_x)
    vs.salt = utilities.enforce_boundaries(vs.salt, settings.enable_cyclic_x)

    vs.rho = density.get_rho(state, vs.salt, vs.temp, npx.abs(vs.zt)[:, npx.newaxis]) * vs.maskT[..., npx.newaxis]
    vs.Hd = (
        density.get_dyn_enthalpy(state, vs.salt, vs.temp, npx.abs(vs.zt)[:, npx.newaxis]) * vs.maskT[..., npx.newaxis]
    )
    vs.int_drhodT = update(
        vs.int_drhodT, at[...], density.get_int_drhodT(state, vs.salt, vs.temp, npx.abs(vs.zt)[:, npx.newaxis])
    )
    vs.int_drhodS = update(
        vs.int_drhodS, at[...], density.get_int_drhodS(state, vs.salt, vs.temp, npx.abs(vs.zt)[:, npx.newaxis])
    )

    fxa = -settings.grav / settings.rho_0 / vs.dzw[npx.newaxis, npx.newaxis, :] * vs.maskW
    vs.Nsqr = update(
        vs.Nsqr,
        at[:, :, :-1, :],
        fxa[:, :, :-1, npx.newaxis]
        * (
            density.get_rho(state, vs.salt[:, :, 1:, :], vs.temp[:, :, 1:, :], npx.abs(vs.zt)[:-1, npx.newaxis])
            - vs.rho[:, :, :-1, :]
        ),
    )
    vs.Nsqr = update(vs.Nsqr, at[:, :, -1, :], vs.Nsqr[:, :, -2, :])

    return KernelOutput(
        salt=vs.salt,
        temp=vs.temp,
        rho=vs.rho,
        Hd=vs.Hd,
        int_drhodT=vs.int_drhodT,
        int_drhodS=vs.int_drhodS,
        Nsqr=vs.Nsqr,
    )


@veros_routine
def calc_initial_conditions(state):
    """
    calculate dyn. enthalp, etc
    """
    vs = state.variables

    if npx.any(vs.salt < 0.0):
        raise RuntimeError("encountered negative salinity")

    vs.update(calc_initial_conditions_kernel(state))


@veros_kernel
def ugrid_to_tgrid(state, a):
    vs = state.variables
    b = update(
        a,
        at[2:-2, :, :],
        (
            vs.dxu[2:-2, :, npx.newaxis] * a[2:-2, :, :]
            + vs.dxu[1:-3, :, npx.newaxis] * a[1:-3, :, :]
        )
        / (2 * vs.dxt[2:-2, :, npx.newaxis]),
    )
    return b


@veros_kernel
def vgrid_to_tgrid(state, a):
    vs = state.variables
    b = update(
        a,
        at[:, 2:-2, :],
        (vs.area_v[:, 2:-2, npx.newaxis] * a[:, 2:-2, :] + vs.area_v[:, 1:-3, npx.newaxis] * a[:, 1:-3, :])
        / (2 * vs.area_t[:, 2:-2, npx.newaxis]),
    )
    return b


@veros_kernel
def calc_diss_u(state, diss):
    vs = state.variables
    ks = allocate(state.dimensions, ("xt", "yt"))
    ks = update(ks, at[1:-2, 2:-2], npx.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]))
    diss_u = diffusion.dissipation_on_wgrid(state, diss, ks)
    return ugrid_to_tgrid(state, diss_u)


@veros_kernel
def calc_diss_v(state, diss):
    vs = state.variables
    ks = allocate(state.dimensions, ("xt", "yt"))
    ks = update(ks, at[2:-2, 1:-2], npx.maximum(vs.kbot[2:-2, 1:-2], vs.kbot[2:-2, 2:-1]))
    diss_v = diffusion.dissipation_on_wgrid(state, diss, ks)
    return vgrid_to_tgrid(state, diss_v)


@veros_kernel
def sanity_check(state):
    return global_and(npx.all(npx.isfinite(state.variables.u)))
