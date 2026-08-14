import os

from veros import VerosSetup, veros_routine, veros_kernel, KernelOutput
from veros.core.operators import numpy as npx, update, at
from veros.variables import Variable

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIP_GRID_FILE = os.path.join(os.path.dirname(BASE_PATH), "DisplacedPoleGrid.SCRIP.nc")

# Number of polar rows masked as land at each pole, chosen from the grid's
# own cell aspect-ratio (max(dxt,dyt)/min(dxt,dyt)) and area (relative to the
# grid median), not just "one row of true/near singularity":
#   south: j=0..3   -- j=3 is the first southward row where aspect ratio
#                       drops back under ~5.5:1 (area ~26% of median);
#                       j=0 alone is 38:1 (~4% of median area).
#   north: j=53..58 -- j=53 is the first northward row where aspect ratio
#                       drops back under ~6:1 (area ~32% of median); j=57 (the
#                       last row masked before this fix) was 18.7:1 (~8%).
# Without this, cells this thin/small are a plausible driver of the
# streamfunction/pressure-solver convergence failures seen around day
# 10-30 of a run, independent of which barotropic solver is used.
SOUTH_LAND_ROWS = 4
NORTH_LAND_ROWS = 6


class GlobalFourDegreeDisplacedPoleSetup(VerosSetup):
    """Curvilinear (displaced-pole) counterpart to the standard global_4deg
    setup, demonstrating the SCRIP-based curvilinear grid path
    (settings.enable_curvilinear_grid) end to end.

    Scope, matching plan-extend-curvlinear.md's Phase 8 recommendation: this
    is an integration/smoke test for the curvilinear grid machinery, not a
    scientifically faithful reproduction of the regular-grid global_4deg
    setup. Topography is an idealized aquaplanet of uniform depth, with a
    band of rows at each pole masked as land (see SOUTH_LAND_ROWS/
    NORTH_LAND_ROWS below) -- wide enough to also exclude the badly
    stretched, small-area cells the grid develops near both poles, not just
    the coordinate singularity itself. Real bathymetry is out of scope here.
    Initial conditions and forcing are likewise idealized (depth-only T/S
    profile, analytic latitude-band wind stress and heat-flux restoring,
    matching the style of the ACC setup) rather than real climatologies --
    regridding real-world forcing onto this grid is separate, deliberately
    deferred work (see the plan).

    The momentum equation's grid-curvature metric term is disabled on this
    grid (see veros/core/momentum.py::tend_coriolisf and
    veros/core/numerics.py::calc_grid_scrip's startup warning) -- a known,
    documented limitation, not a bug.
    """

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = "global_4deg_displaced_pole"
        settings.description = "Curvilinear displaced-pole counterpart to global_4deg"

        settings.nx, settings.ny, settings.nz = 120, 59, 15
        settings.dt_mom = 1800.0
        settings.dt_tracer = 86400.0
        settings.runlen = 0.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True
        settings.enable_curvilinear_grid = True
        settings.scrip_grid_file = SCRIP_GRID_FILE

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 1000.0
        settings.iso_dslope = 4.0 / 1000.0
        settings.iso_slopec = 1.0 / 1000.0
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = (4 * settings.degtom) ** 3 * 2e-11
        # cos-scaling of horizontal viscosity assumes a regular lat-lon grid
        # (see friction.py's use of cost/cosu); leave it off here since
        # cost=cosu=1 on the curvilinear path (calc_grid_scrip) would make it
        # a no-op anyway
        settings.enable_hor_friction_cos_scaling = False

        settings.enable_implicit_vert_friction = True
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True

        settings.enable_idemix = False

        settings.eq_of_state_type = 5

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("xt", "yt"), "deg C", "Reference surface temperature"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        # horizontal grid comes entirely from settings.scrip_grid_file via
        # calc_grid_scrip; only the vertical grid is set here
        ddz = npx.array(
            [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
        )
        vs.dzt = ddz[::-1]

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        # yt is already true 2D geographic latitude (populated by
        # calc_grid_scrip), so this is unchanged from the regular-grid
        # setups' own set_coriolis other than not needing a 1D broadcast
        vs.coriolis_t = update(vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt / 180.0 * settings.pi))

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        # idealized aquaplanet of uniform depth (full water column, kbot=1
        # everywhere) except a band of rows at each pole (SOUTH_LAND_ROWS,
        # NORTH_LAND_ROWS above), masked as land -- wide enough to clear both
        # the coordinate singularity and the badly stretched cells nearby.
        # See this setup's docstring for why real bathymetry is out of scope.
        ocean = npx.ones_like(vs.kbot)
        ocean = update(ocean, at[:, : 2 + SOUTH_LAND_ROWS], 0)
        ocean = update(ocean, at[:, 2 + settings.ny - NORTH_LAND_ROWS :], 0)
        vs.kbot = ocean

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        # idealized initial conditions: depth-dependent T/S profile only,
        # matching the style of the ACC setup -- see this file's docstring
        # for why real climatology regridding is out of scope here
        vs.temp = update(
            vs.temp, at[...], ((1 - vs.zt[npx.newaxis, npx.newaxis, :] / vs.zw[0]) * 20 * vs.maskT)[..., npx.newaxis]
        )
        vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., npx.newaxis])

        # analytic latitude-band wind stress and restoring temperature,
        # operating directly on the (now genuinely 2D) yt/yu -- no change in
        # form from a regular-grid idealized setup, since yt/yu already hold
        # true geographic latitude everywhere
        yt_min = float(npx.min(vs.yt))
        yt_max = float(npx.max(vs.yt))

        taux = npx.zeros_like(vs.yt)
        taux = npx.where(vs.yt < -20, 0.1 * npx.sin(settings.pi * (vs.yt - yt_min) / (-20.0 - yt_min)), taux)
        taux = npx.where(vs.yt > 10, 0.1 * (1 - npx.cos(2 * settings.pi * (vs.yt - 10.0) / (yt_max - 10.0))), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        vs.t_star = npx.where(vs.yt < -20, 20 * (vs.yt - yt_min) / (-20 - yt_min), 20.0)
        vs.t_star = npx.where(vs.yt > 20, 20 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
        vs.t_rest = vs.dzt[npx.newaxis, npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.update(set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        state.diagnostics["snapshot"].output_frequency = 360 * 86400.0
        state.diagnostics["energy"].output_frequency = 360 * 86400.0
        state.diagnostics["energy"].sampling_frequency = 86400
        average_vars = ["temp", "salt", "u", "v", "w", "surface_taux", "surface_tauy", "psi"]
        state.diagnostics["averages"].output_variables = average_vars
        state.diagnostics["averages"].output_frequency = 360 * 86400.0
        state.diagnostics["averages"].sampling_frequency = 86400
        # the overturning diagnostic bins transport by j-index, assuming
        # constant-j = constant-latitude -- false on this grid (out of scope,
        # see plan-extend-curvlinear.md's Context section); leave it off
        state.diagnostics["overturning"].output_frequency = 0.0

    @veros_routine
    def after_timestep(self, state):
        pass


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
            )
            ** 1.5,
        )

    vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    return KernelOutput(forc_tke_surface=vs.forc_tke_surface, forc_temp_surface=vs.forc_temp_surface)
