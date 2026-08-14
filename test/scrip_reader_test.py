import os

import numpy as np
import pytest

earthsystemgrids = pytest.importorskip("EarthSystemGrids", reason="requires the optional earthsystemgrids package")

from veros.tools import read_scrip_grid, regular_grid_to_scrip  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISPLACED_POLE_GRID_FILE = os.path.join(REPO_ROOT, "DisplacedPoleGrid.SCRIP.nc")

EARTH_RADIUS = 6370e3
EARTH_SURFACE_AREA = 4 * np.pi * EARTH_RADIUS**2


@pytest.mark.skipif(not os.path.isfile(DISPLACED_POLE_GRID_FILE), reason="DisplacedPoleGrid.SCRIP.nc not present")
def test_displaced_pole_grid_sanity():
    grid = read_scrip_grid(DISPLACED_POLE_GRID_FILE, radius=EARTH_RADIUS)

    assert grid.shape == (120, 59)

    for name in ("xt", "yt", "xu", "yu", "dxt", "dyt", "dxu", "dyu", "area_t"):
        arr = getattr(grid, name)
        assert arr.shape == grid.shape
        assert np.isfinite(arr).all(), f"{name} contains non-finite values"

    assert -90.0 <= grid.yt.min() and grid.yt.max() <= 90.0
    assert -180.0 <= grid.xt.min() and grid.xt.max() <= 180.0
    assert -90.0 <= grid.yu.min() and grid.yu.max() <= 90.0
    assert -180.0 <= grid.xu.min() and grid.xu.max() <= 180.0

    # cell widths/heights and area must be strictly positive everywhere
    # except the pole-closing ring, where the grid intentionally collapses
    # to (near-)zero width by construction
    assert (grid.dyt > 0).all()
    assert (grid.area_t >= 0).all()
    near_zero = grid.dxt < 1.0
    assert near_zero.sum() <= grid.shape[0], "unexpectedly many near-zero-width cells"

    # total area should closely match Earth's true surface area
    rel_error = abs(grid.area_t.sum() - EARTH_SURFACE_AREA) / EARTH_SURFACE_AREA
    assert rel_error < 1e-2

    assert grid.mask.min() >= 0


def test_regular_grid_round_trip(tmp_path):
    nx, ny = 20, 15
    dlon, dlat = 4.0, 4.0
    lon_origin, lat_origin = 4.0, -76.0

    scrip_file = str(tmp_path / "regular.scrip.nc")
    regular_grid_to_scrip(scrip_file, nx=nx, ny=ny, dlon=dlon, dlat=dlat, lon_origin=lon_origin, lat_origin=lat_origin)

    grid = read_scrip_grid(scrip_file, radius=EARTH_RADIUS)
    assert grid.shape == (nx, ny)

    # dyt does not depend on cos(latitude), so it must match the analytic
    # value exactly everywhere
    dlat_m = np.deg2rad(dlat) * EARTH_RADIUS
    np.testing.assert_allclose(grid.dyt, dlat_m, rtol=1e-8)

    # dxt should match the analytic cos(latitude)-scaled value closely
    dlon_m = np.deg2rad(dlon) * EARTH_RADIUS * np.cos(np.deg2rad(grid.yt))
    np.testing.assert_allclose(grid.dxt, dlon_m, rtol=1e-2)

    assert (grid.mask == 1).all()

    # analytic area of each cell: R^2 * dlon_rad * dlat_rad * cos(lat) -- using
    # the grid's own actual cell latitudes rather than a nominal lon/lat box
    # derived from lon_origin/lat_origin, since the underlying legacy grid
    # convention (matched via u_centered_grid, see regular_grid_to_scrip)
    # anchors the origin to a U-point, not the domain's outer edge, so the
    # true coverage is offset by a fraction of a cell from that nominal box
    expected_area = (EARTH_RADIUS**2 * np.deg2rad(dlon) * np.deg2rad(dlat) * np.cos(np.deg2rad(grid.yt))).sum()
    rel_error = abs(grid.area_t.sum() - expected_area) / expected_area
    assert rel_error < 1e-2
