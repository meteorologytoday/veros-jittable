"""
Adapter for loading a curvilinear horizontal grid from a SCRIP-format netCDF
file, via the `earthsystemgrids` library (which is what generates such files
in the first place -- see EarthSystemGrids.base.StructuredQuadMesh).

This module only *projects* the library's mesh object onto Veros's
(nx, ny)-shaped Arakawa-C grid arrays (xt/yt/xu/yu/dxt/dyt/dxu/dyu/area_t/
mask); all spherical geometry (great-circle lengths, node/edge connectivity)
is delegated to the library, not reimplemented here.

Kept independent of veros.core/veros.variables/VerosState, so it can be unit
tested without a full model setup -- see test/scrip_reader_test.py.

`earthsystemgrids` is an optional dependency: only importing this specific
module (or calling its functions) requires it, so plain `import veros` /
`import veros.tools` keeps working for anyone who never touches curvilinear
grids.
"""

import numpy as onp


def _earthsystemgrids():
    try:
        from EarthSystemGrids.base.UnstructuredGridMesh import _R_EARTH, _great_circle_lengths
        from EarthSystemGrids.base.StructuredQuadMesh import StructuredQuadMesh
    except ImportError as e:
        raise ImportError(
            "Reading/writing SCRIP grid files requires the 'earthsystemgrids' package "
            "(https://github.com/meteorologytoday/EarthSystemGrids), which is not installed."
        ) from e
    return _R_EARTH, _great_circle_lengths, StructuredQuadMesh


class CurvilinearGrid:
    """
    Plain container for the Arakawa-C grid arrays derived from a SCRIP file.

    All arrays are shaped (nx, ny) -- axis 0 is the i/x-direction, axis 1 is
    the j/y-direction, matching Veros's array-index convention -- and carry
    no ghost cells (that padding is applied by the caller, e.g.
    veros.core.numerics.calc_grid_scrip, the same way the legacy analytic
    path pads calc_grid_spacings_kernel's output).

    xt/yt/xu/yu are in degrees. dxt/dyt/dxu/dyu/area_t are scaled to whatever
    `radius` was passed to `read_scrip_grid`/`regular_grid_to_scrip`, so they
    stay consistent with the rest of Veros's physics (which uses
    settings.radius throughout).

    xu is the true longitude of the U-point (shared edge between T(i,j) and
    T(i+1,j)); yu is the true latitude of the V-point (shared edge between
    T(i,j) and T(i,j+1)). There is no separate "true latitude of U-point" or
    "true longitude of V-point" here, matching the fact that Veros's Variable
    set has no slot for either -- U_HOR=(xu,yt) and V_HOR=(xt,yu) only ever
    needed xu and yu themselves on the legacy separable grid.
    """

    def __init__(self, xt, yt, xu, yu, dxt, dyt, dxu, dyu, area_t, mask):
        self.xt = xt
        self.yt = yt
        self.xu = xu
        self.yu = yu
        self.dxt = dxt
        self.dyt = dyt
        self.dxu = dxu
        self.dyu = dyu
        self.area_t = area_t
        self.mask = mask

    @property
    def shape(self):
        return self.xt.shape


def _lon_midpoint(lon1, lon2):
    """Average of two nearby longitudes (degrees), robust to +-180 wraparound."""
    diff = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    return lon1 + diff / 2.0


def _mesh_to_curvilinear_grid(mesh, radius):
    """
    Derive Veros's (nx, ny)-shaped Arakawa-C grid arrays from an
    EarthSystemGrids StructuredQuadMesh's structured (nj, ni) node/edge
    connectivity (see EarthSystemGrids.base.StructuredQuadMesh._build_topology_2d
    for the corner/edge ordering this relies on: corners are
    [SW, SE, NE, NW] and edges are [bottom, right, top, left]).
    """
    _R_EARTH, _great_circle_lengths, _ = _earthsystemgrids()

    nj, ni = mesh.shape
    rad2deg = 180.0 / onp.pi
    # the library's edge_length/area are computed with its own fixed internal
    # radius (_R_EARTH); rescale to whatever radius the caller actually wants
    # (settings.radius), which need not be identical
    scale = radius / _R_EARTH

    def to_ij(flat):
        return onp.asarray(flat).reshape(nj, ni).T

    face_nodes = mesh.face_nodes.reshape(nj, ni, 4).transpose(1, 0, 2)  # (i, j, corner)
    face_edges = mesh.face_edges.reshape(nj, ni, 4).transpose(1, 0, 2)  # (i, j, edge)

    xt = to_ij(mesh.face_lon) * rad2deg
    yt = to_ij(mesh.face_lat) * rad2deg
    area_t = to_ij(mesh.area) * scale**2
    mask = to_ij(mesh.mask)

    edge_length = onp.asarray(mesh.edge_length)
    # bottom(0)/top(2) edges run in the i-direction -> T-cell width;
    # right(1)/left(3) edges run in the j-direction -> T-cell height
    dxt = 0.5 * (edge_length[face_edges[:, :, 0]] + edge_length[face_edges[:, :, 2]]) * scale
    dyt = 0.5 * (edge_length[face_edges[:, :, 1]] + edge_length[face_edges[:, :, 3]]) * scale

    node_lon = onp.asarray(mesh.node_lon) * rad2deg
    node_lat = onp.asarray(mesh.node_lat) * rad2deg

    # U-point: midpoint of the shared (right) edge between T(i,j) and
    # T(i+1,j) -- connects T(i,j)'s SE (corner 1) and NE (corner 2) nodes
    se, ne, nw = face_nodes[:, :, 1], face_nodes[:, :, 2], face_nodes[:, :, 3]
    xu = _lon_midpoint(node_lon[se], node_lon[ne])
    # near a degenerate cell (e.g. the pole-closing ring of a displaced-pole
    # grid, where cell width collapses to ~0 and "the" longitude is barely
    # meaningful), the two corners can legitimately be >180 deg apart, and
    # the wraparound-averaged result can land outside the canonical range --
    # wrap it back rather than propagate an out-of-range longitude downstream
    xu = ((xu + 180.0) % 360.0) - 180.0

    # V-point: midpoint of the shared (top) edge between T(i,j) and T(i,j+1)
    # -- connects T(i,j)'s NE (corner 2) and NW (corner 3) nodes
    yu = 0.5 * (node_lat[ne] + node_lat[nw])

    # dxu/dyu: great-circle distance between adjacent T-point centres -- this
    # matches the legacy 1D u_centered_grid's definition (dyu = yt[1:]-yt[:-1]
    # there), not the cell edge length. The last row/column has no i+1/j+1
    # neighbour; reuse the previous spacing, same as calc_grid_spacings_kernel
    # does for its ghost cells.
    xt_r, yt_r = onp.deg2rad(xt), onp.deg2rad(yt)

    dxu = onp.empty_like(xt)
    dxu[:-1, :] = _great_circle_lengths(xt_r[:-1, :], yt_r[:-1, :], xt_r[1:, :], yt_r[1:, :]) * scale
    dxu[-1, :] = dxu[-2, :]

    dyu = onp.empty_like(yt)
    dyu[:, :-1] = _great_circle_lengths(xt_r[:, :-1], yt_r[:, :-1], xt_r[:, 1:], yt_r[:, 1:]) * scale
    dyu[:, -1] = dyu[:, -2]

    return CurvilinearGrid(xt=xt, yt=yt, xu=xu, yu=yu, dxt=dxt, dyt=dyt, dxu=dxu, dyu=dyu, area_t=area_t, mask=mask)


def read_scrip_grid(scrip_file, radius):
    """
    Load a curvilinear horizontal grid from a SCRIP-format netCDF file.

    Uses `earthsystemgrids.StructuredQuadMesh.from_SCRIP_file` to parse the
    file and build its node/edge connectivity; this function only projects
    that mesh onto Veros's (nx, ny)-shaped Arakawa-C grid arrays. No new
    spherical-geometry math is done here.

    Arguments:
        scrip_file: path to a SCRIP-format grid file.
        radius: sphere radius in metres to scale lengths/areas to -- pass
            settings.radius so grid metrics stay consistent with the rest of
            Veros's physics.

    Returns:
        A CurvilinearGrid with every array shaped (nx, ny), no ghost cells.
    """
    _, _, StructuredQuadMesh = _earthsystemgrids()
    mesh = StructuredQuadMesh.from_SCRIP_file(scrip_file)
    return _mesh_to_curvilinear_grid(mesh, radius)


def regular_grid_to_scrip(output_file, nx, ny, dlon, dlat, lon_origin, lat_origin):
    """
    Write a SCRIP-format grid file describing a regular lat-lon grid with
    uniform spacing `dlon`/`dlat` and the given origin -- i.e. the same grid
    a Veros setup gets from `coord_degree=True`, uniform `dxt=dlon`,
    `dyt=dlat`, `x_origin=lon_origin`, `y_origin=lat_origin` on the existing
    (legacy, analytic) grid path.

    This is the fixture every curvilinear-path regression gate depends on:
    read this file back through `read_scrip_grid` and the result should
    match `veros.core.numerics.calc_grid_spacings_kernel`'s output for the
    same regular grid to floating-point tolerance (see
    test/scrip_reader_test.py and test/curvilinear_grid_test.py). It is not
    meant to describe a real curvilinear grid.
    """
    _R_EARTH, _, _ = _earthsystemgrids()

    # Reuse the legacy grid kernel's own 1D profile construction rather than
    # re-deriving its cell-center/edge alignment convention independently --
    # calc_grid_spacings_kernel anchors x_origin/y_origin to the *U-point* at
    # ghost-index 2 (a cell edge), not the T-point center, which a naive
    # "origin + (i+0.5)*d" formula would get subtly wrong (off by half a
    # cell). Importing here, not at module level, to avoid pulling all of
    # veros.core into this module's plain import path.
    from veros.core.numerics import u_centered_grid

    dxt_1d = dlon * onp.ones(nx + 4)
    _, xt_1d, xu_1d = u_centered_grid(dxt_1d, onp.zeros(nx + 4), onp.zeros(nx + 4), onp.zeros(nx + 4))
    xt_1d = xt_1d + lon_origin - xu_1d[2]
    xu_1d = xu_1d + lon_origin - xu_1d[2]

    dyt_1d = dlat * onp.ones(ny + 4)
    _, yt_1d, yu_1d = u_centered_grid(dyt_1d, onp.zeros(ny + 4), onp.zeros(ny + 4), onp.zeros(ny + 4))
    yt_1d = yt_1d + lat_origin - yu_1d[2]
    yu_1d = yu_1d + lat_origin - yu_1d[2]

    lon_t_1d = onp.asarray(xt_1d[2:-2])
    lat_t_1d = onp.asarray(yt_1d[2:-2])
    lon_t = onp.broadcast_to(lon_t_1d[:, None], (nx, ny))
    lat_t = onp.broadcast_to(lat_t_1d[None, :], (nx, ny))

    # T-cell i's west/east edges are U-points xu_1d[i+1]/xu_1d[i+2] (U-point
    # index i sits east of T-point index i, standard Arakawa-C convention;
    # +2 to account for the two leading ghost cells in xu_1d/yu_1d)
    lon_edge_1d = onp.asarray(xu_1d[1:-2])  # length nx+1: west edges of cells 0..nx-1, plus the final east edge
    lat_edge_1d = onp.asarray(yu_1d[1:-2])

    # SW, SE, NE, NW corners of cell (i, j), matching
    # EarthSystemGrids' StructuredQuadMesh corner-ordering convention
    corner_lon = onp.empty((nx, ny, 4))
    corner_lat = onp.empty((nx, ny, 4))
    for k, (di, dj) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        corner_lon[:, :, k] = onp.broadcast_to(lon_edge_1d[di : di + nx][:, None], (nx, ny))
        corner_lat[:, :, k] = onp.broadcast_to(lat_edge_1d[dj : dj + ny][None, :], (nx, ny))

    area = onp.deg2rad(dlon) * onp.deg2rad(dlat) * onp.cos(onp.deg2rad(lat_t)) * _R_EARTH**2
    mask = onp.ones((nx, ny), dtype="int32")

    import xarray as xr

    ds = xr.Dataset(
        data_vars=dict(
            grid_dims=(["grid_rank"], [nx, ny]),
            grid_imask=(["grid_size"], mask.T.ravel()),
            grid_center_lat=(["grid_size"], lat_t.T.ravel(), {"units": "degrees"}),
            grid_center_lon=(["grid_size"], lon_t.T.ravel(), {"units": "degrees"}),
            grid_corner_lat=(["grid_size", "grid_corners"], corner_lat.transpose(1, 0, 2).reshape(-1, 4), {"units": "degrees"}),
            grid_corner_lon=(["grid_size", "grid_corners"], corner_lon.transpose(1, 0, 2).reshape(-1, 4), {"units": "degrees"}),
            grid_area=(["grid_size"], (area / _R_EARTH**2).T.ravel(), {"units": "radians^2"}),
        ),
    )
    # deliberately not pinning `engine=` here: EarthSystemGrids.StructuredQuadMesh
    # .from_SCRIP_file reads this back via xr.open_dataset with no engine
    # pinned either, and xarray's dimension-scale metadata differs subtly
    # between its h5netcdf and netCDF4 backends -- writing and reading with
    # mismatched engines produces spurious HDF5 errors, so leave both sides
    # on xarray's own (consistent) default engine resolution.
    ds.to_netcdf(output_file)
    return output_file
