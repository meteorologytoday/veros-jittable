"""
Verifies the curvilinear grid-init path (calc_grid_scrip) reproduces the
legacy analytic path (calc_grid_spacings_kernel + calc_grid_metrics_kernel)
on a SCRIP description of the *same* regular grid -- the standing invariant
this whole feature is built around (see plan-extend-curvlinear.md).
"""

import os

import numpy as np
import pytest

pytest.importorskip("EarthSystemGrids", reason="requires the optional earthsystemgrids package")

from veros.state import get_default_state  # noqa: E402
from veros.core import numerics  # noqa: E402
from veros.tools import regular_grid_to_scrip  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPLACED_POLE_GRID_FILE = os.path.join(REPO_ROOT, "DisplacedPoleGrid.SCRIP.nc")


NX, NY, NZ = 90, 40, 2
DLON, DLAT = 4.0, 4.0
LON_ORIGIN, LAT_ORIGIN = 4.0, -76.0


def _make_legacy_state():
    state = get_default_state()
    with state.settings.unlock():
        s = state.settings
        s.nx, s.ny, s.nz = NX, NY, NZ
        s.dt_mom, s.dt_tracer = 1800.0, 86400.0
        s.x_origin, s.y_origin = LON_ORIGIN, LAT_ORIGIN
        s.coord_degree = True
        s.enable_cyclic_x = True
    state.initialize_variables()
    vs = state.variables
    with vs.unlock():
        vs.dxt = DLON * np.ones_like(vs.dxt)
        vs.dyt = DLAT * np.ones_like(vs.dyt)
        vs.dzt = np.array([100.0, 150.0])
    numerics.calc_grid(state)
    return state


def _make_curvilinear_state(scrip_file):
    state = get_default_state()
    with state.settings.unlock():
        s = state.settings
        s.nx, s.ny, s.nz = NX, NY, NZ
        s.dt_mom, s.dt_tracer = 1800.0, 86400.0
        s.coord_degree = True
        s.enable_cyclic_x = True
        s.enable_curvilinear_grid = True
        s.scrip_grid_file = scrip_file
    state.initialize_variables()
    vs = state.variables
    with vs.unlock():
        vs.dzt = np.array([100.0, 150.0])
    numerics.calc_grid(state)
    return state


@pytest.fixture(scope="module")
def grid_states(tmp_path_factory):
    scrip_file = str(tmp_path_factory.mktemp("scrip") / "regular.scrip.nc")
    regular_grid_to_scrip(scrip_file, nx=NX, ny=NY, dlon=DLON, dlat=DLAT, lon_origin=LON_ORIGIN, lat_origin=LAT_ORIGIN)
    return _make_legacy_state(), _make_curvilinear_state(scrip_file)


def _interior(vs, name):
    return np.asarray(getattr(vs, name))[2:-2, 2:-2]


def _lon_diff(a, b):
    """Difference between two longitude arrays, accounting for the 360-degree
    periodic ambiguity (both representations are physically equivalent)."""
    return (a - b + 180.0) % 360.0 - 180.0


def test_positions_match(grid_states):
    legacy, curv = grid_states

    np.testing.assert_allclose(_lon_diff(_interior(legacy.variables, "xt"), _interior(curv.variables, "xt")), 0.0, atol=1e-8)
    np.testing.assert_allclose(_interior(legacy.variables, "yt"), _interior(curv.variables, "yt"), atol=1e-8)
    np.testing.assert_allclose(_lon_diff(_interior(legacy.variables, "xu"), _interior(curv.variables, "xu")), 0.0, atol=1e-8)
    np.testing.assert_allclose(_interior(legacy.variables, "yu"), _interior(curv.variables, "yu"), atol=1e-8)


def test_meridional_spacing_matches_exactly(grid_states):
    # dyt/dyu never depend on cos(latitude), so both paths must agree almost
    # to the bit (up to a few ULPs of float64 accumulation through cumsum)
    legacy, curv = grid_states
    np.testing.assert_allclose(_interior(legacy.variables, "dyt"), _interior(curv.variables, "dyt"), atol=1e-6)
    np.testing.assert_allclose(_interior(legacy.variables, "dyu"), _interior(curv.variables, "dyu"), atol=1e-6)


def test_zonal_spacing_matches_true_distance(grid_states):
    # the legacy dxt/dxu are degtom-scaled *degrees*, not true distances --
    # cost/cosu=1 on the curvilinear path is the resolution to that (see
    # calc_grid_scrip's docstring), so compare against the legacy path's own
    # cost/cosu-corrected true distance, not raw dxt/dxu.
    #
    # dxu is a T-to-T distance that lives at the U-point, whose true latitude
    # is yt (U_HOR=(xu,yt) -- U-points don't shift latitude on a regular
    # grid), so it's corrected by *cost*, not cosu -- matching how dxu is
    # paired with cost, not cosu, everywhere it's consumed downstream (e.g.
    # veros/core/external/poisson_matrix.py's hu/dxu/dxt/cost**2 term).
    legacy, curv = grid_states
    lv, cv = legacy.variables, curv.variables

    dxt_true_legacy = _interior(lv, "dxt") * _interior(lv, "cost")
    np.testing.assert_allclose(dxt_true_legacy, _interior(cv, "dxt"), rtol=2e-3)

    dxu_true_legacy = _interior(lv, "dxu") * _interior(lv, "cost")
    np.testing.assert_allclose(dxu_true_legacy, _interior(cv, "dxu"), rtol=2e-3)


def test_area_matches(grid_states):
    legacy, curv = grid_states
    # area_t is exact (both paths use the SCRIP file's own precise
    # spherical-polygon area); area_u/area_v propagate dxu/dyu's small
    # edge-averaging-vs-analytic discretization difference, same order as
    # test_zonal_spacing_matches_true_distance's tolerance
    np.testing.assert_allclose(_interior(legacy.variables, "area_t"), _interior(curv.variables, "area_t"), rtol=1e-6)
    np.testing.assert_allclose(_interior(legacy.variables, "area_u"), _interior(curv.variables, "area_u"), rtol=2e-3)

    # area_v needs a cell width evaluated at the V-point's own (j-shifted)
    # latitude, approximated in calc_grid_scrip by averaging the two T-cells
    # straddling it in j -- that average has no "next" T-cell at the very
    # last row, so calc_grid_scrip falls back to reusing that row's own
    # value there. Check the interior tightly and only sanity-check (not
    # tightly match) the one known boundary row.
    area_v_legacy = _interior(legacy.variables, "area_v")
    area_v_curv = _interior(curv.variables, "area_v")
    np.testing.assert_allclose(area_v_legacy[:, :-1], area_v_curv[:, :-1], rtol=2e-3)
    assert area_v_curv[:, -1].min() > 0


def test_curvilinear_grid_shape_check_rejects_mismatched_nx_ny(tmp_path):
    scrip_file = str(tmp_path / "regular.scrip.nc")
    regular_grid_to_scrip(scrip_file, nx=NX, ny=NY, dlon=DLON, dlat=DLAT, lon_origin=LON_ORIGIN, lat_origin=LAT_ORIGIN)

    state = get_default_state()
    with state.settings.unlock():
        s = state.settings
        s.nx, s.ny, s.nz = NX + 1, NY, NZ  # deliberately wrong
        s.coord_degree = True
        s.enable_curvilinear_grid = True
        s.scrip_grid_file = scrip_file
    state.initialize_variables()

    with pytest.raises(ValueError):
        numerics.calc_grid(state)


@pytest.mark.skipif(not os.path.isfile(DISPLACED_POLE_GRID_FILE), reason="DisplacedPoleGrid.SCRIP.nc not present")
def test_displaced_pole_smoke_run():
    """
    Phase 8's "real displaced-pole smoke test": run the idealized
    aquaplanet setup on the actual DisplacedPoleGrid.SCRIP.nc grid for a
    few days and check the result is finite and physically plausible.

    This is deliberately a short run (5 tracer days) -- longer integrations
    on this setup are known to drift and eventually fail to converge,
    consistent with the deferred momentum metric term (Phase 6) leaving the
    curvilinear path without a proper Coriolis/curvature balance. That's a
    documented limitation of this delivery, not something this test is
    meant to catch; it verifies the grid and machinery work end-to-end, not
    long-term stability.
    """
    import sys

    from veros import runtime_settings
    from veros.core.operators import numpy as npx

    sys.path.insert(0, REPO_ROOT)
    from test_case_displaced_pole_grid.global_4deg import GlobalFourDegreeDisplacedPoleSetup

    object.__setattr__(runtime_settings, "diskless_mode", True)
    try:
        setup = GlobalFourDegreeDisplacedPoleSetup()
        setup.setup()

        with setup.state.settings.unlock():
            setup.state.settings.runlen = 5 * setup.state.settings.dt_tracer

        setup.run()
    finally:
        object.__setattr__(runtime_settings, "diskless_mode", False)

    vs = setup.state.variables

    for name in ("temp", "salt", "u", "v", "w", "psi"):
        arr = npx.asarray(getattr(vs, name))
        assert npx.isfinite(arr).all(), f"{name} contains non-finite values after the smoke run"

    assert -5.0 <= float(vs.temp.min()) and float(vs.temp.max()) <= 40.0
    assert 0.0 <= float(vs.salt.min()) and float(vs.salt.max()) <= 45.0
    # SOUTH_LAND_ROWS + NORTH_LAND_ROWS out of ny=59 rows are land
    ocean_frac = float((npx.asarray(vs.kbot)[2:-2, 2:-2] > 0).mean())
    assert 0.75 < ocean_frac < 0.9
