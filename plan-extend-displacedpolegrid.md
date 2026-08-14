# Curvilinear (displaced-pole) grid support for Veros

## Context

Veros currently assumes a regular, fully-separable lat-lon horizontal grid: `xt`, `yt`, `dxt`, `dyt`, and the metric factors `cost = cos(lat)`, `cosu`, `tantr = tan(lat)/R` are all 1D arrays (functions of `i` or `j` alone), and the momentum equation's curvature/metric term is the spherical-coordinate formula `tan(lat)/R`. The goal is to run the model on a POP-style **displaced-pole** grid (described by `DisplacedPoleGrid.SCRIP.nc`), where `i` is not zonal and `j` is not meridional everywhere — the pole is moved over land specifically to avoid the coordinate singularity and CFL-limiting grid convergence of a true lat-lon grid at the pole.

Scope, as agreed: target **locally-orthogonal curvilinear grids only** (diagonal metric tensor, no cross terms) — matches how displaced-pole/conformal grids are actually constructed, not a fully general non-orthogonal framework.

The momentum-equation curvature term (`tantr = tan(lat)/R`) is **excluded from this plan's scope entirely**, by explicit decision — not derived, not dropped-and-approved, just deferred. It's the one place in the whole project needing a careful, reviewed physical derivation (the general orthogonal-curvilinear form, a function of local scale factors and their gradients, reducing to `tan(lat)/R` only in the lat-lon limit), and that derivation is judged high-risk enough to warrant its own dedicated engineering task rather than riding along with the grid-generalization work here. This plan's only obligation regarding that term is: don't leave it silently wrong. On the curvilinear path, its computation is explicitly commented out (code kept in place, not deleted, for whoever picks up the follow-on task) rather than executed with a formula that's known to be incorrect off a regular lat-lon grid. The legacy analytic path is completely untouched — `tantr` keeps running exactly as it does today, so existing regression tests keep passing unchanged.

Test target: `test_case_displaced_pole_grid/global_4deg.py` (currently an unmodified copy of the regular-grid `global_4deg` setup) becomes the curvilinear integration test, run against `DisplacedPoleGrid.SCRIP.nc` (nx=120, ny=59, confirmed via `grid_dims`).

**Standing invariant, used wherever a phase introduces a new SCRIP-consuming code path:** a SCRIP file describing the *existing* regular lat-lon grid must, run through that new path, reproduce the current 1D/analytic path's output to floating-point tolerance. This applies concretely to Phase 4 (per-array) and Phase 8 (end-to-end) — the two phases that actually exercise the SCRIP fixture. Phases that only touch the legacy analytic path (3, 5, 6) are instead gated by the existing pyOM-consistency/setup regression suites, which is the correct and sufficient check for them since they don't introduce a new grid-consuming path to compare against.

## Key facts established during research (not re-derived at implementation time)

- `veros/variables.py` already has 2D-horizontal dimension-tuple constants: `T_HOR=("xt","yt")`, `U_HOR=("xu","yt")`, `V_HOR=("xt","yu")`, `ZETA_HOR=("xu","yu")` — already used by `coriolis_t`, `kbot`, `area_t/u/v`. Promoting `xt/yt/xu/yu/dxt/dyt/dxu/dyu/cost/cosu/tantr` to these dims is the pattern to follow, not new design.
- `veros/state.py` (Variable allocation) and `veros/distributed.py` (MPI gather/scatter) are **fully dims-tuple-driven, not rank-driven** — confirmed `coriolis_t`/`kbot`/`area_t` already flow through the 2D-horizontal MPI path today. No changes needed in either file.
- `Variable.dims` is a static attribute (`veros/variables.py` `Variable.__init__`, confirmed) — it cannot be conditional on settings. The 1D→2D promotion is global across every setup, not opt-in per setup. "Additive" therefore means *regression-tested to be behavior-preserving*, not *zero shared code*.
- **Two confirmed bugs in `veros/io_tools/netcdf.py::initialize_file`, not a fundamental naming collision** — verified directly: `h5netcdf` (the library Veros uses) and `xarray` both tolerate a variable sharing a name with one of its own dimensions even when that variable is >1D (tested: create/write/read-back round-trips cleanly). So `xt`/`yt`/`xu`/`yu` do **not** need renaming. The real bugs, both artifacts of the loop's implicit assumption that a dimension's namesake variable is 1D: (1) `initialize_file:79`, `dimsize = get_shape(dimensions, var.dims[::-1], ...)[0]` — for 1D `xt` today this correctly gives `nx`; once `xt`'s dims become `T_HOR=("xt","yt")` (2D), `var.dims[::-1] = ("yt","xt")` and `[0]` silently returns `yt`'s size instead, mis-sizing the netCDF `xt` dimension. (2) `initialize_variable:98`, `dims = tuple(d for d in var.dims if d in ncfile.dimensions)` — dimensions are registered one at a time in the same loop that creates variables, so if `yt` hasn't been registered yet when 2D `xt` is processed, the variable gets created with only its `xt` dim while the actual data is 2D, a shape mismatch on write. Fix: split into two passes — register every netCDF dimension first (sized directly from `state.dimensions[dim]`), then create variables in a second pass, once every dimension they need already exists. `xt`/`yt`/`xu`/`yu` keep their current names and stay coordinate-system-neutral throughout (meters for Cartesian setups, degrees for lat-lon ones), consistent with how `dxt`/`dyt` are already documented.
- Elliptic solvers (`poisson_matrix.py`, `solve_stream.py`, `streamfunction_init.py`) build a **fixed 5-point stencil** (static offsets `[(0,0),(1,0),(-1,0),(0,1),(0,-1)]`) from `dxt/dxu/dyt/dyu/cost/cosu` via 1D-broadcast expressions. Generalizing is a **mechanical value substitution** (2D-indexed reads instead of 1D-broadcast) — no change to stencil structure, matrix assembly, or the three solver backends (`scipy.py`/`scipy_jax.py`/`petsc_.py`, which are pure consumers of the diagonal/offset representation and don't reference grid metrics at all).
- `friction.py` (1026 lines) and `isoneutral/{isoneutral,diffusion}.py`: confirmed by full read — every `cost`/`cosu`/`dxt`/`dxu`/`dyt`/`dyu` use is pure multiplicative/divisive scaling (including the `hor_friction_cosPower` exponent term). **Mechanical**, no hidden spherical-specific formulas.
- **The only genuinely spherical-specific, non-scaling formula anywhere in the dynamical core is `tantr`**, used in exactly 3 expressions in `momentum.py:56-91` (`tend_coriolisf`'s `du_cor`/`dv_cor` metric-term additions), computed in `numerics.py:127`. It does not appear in `friction.py` (confirmed via grep) or anywhere else.
- Out of scope, by explicit decision: wind-stress vector rotation (forcing is assumed to already be grid-aligned) and the overturning diagnostic's latitude-binning (not needed). Neither is built.
- No SCRIP/netCDF grid reader exists in `veros/tools/` today; `h5netcdf` is used ad hoc inside setups (e.g. `global_4deg.py::_read_forcing`) — that's the pattern to follow for a new reader.

## Phase 0 — Lock in conventions (no code)

- Keep `dxt/dyt/dxu/dyu/cost/cosu/tantr/area_*/xt/yt/xu/yu` names as-is, in both meaning and identity — reinterpret docstrings as "local grid position/metric factor in the i(t/u)- or j(t/u)-direction," not literally zonal/meridional or geographic. This keeps Veros usable for genuinely Cartesian setups (`xt`/`yt` in metres) exactly as today; on a curvilinear setup they hold true geographic lon/lat (populated from SCRIP `grid_center_lon/lat` and corner-derived U/V/Z points), same variables, no parallel/renamed set.
- Add `enable_curvilinear_grid` (bool, default `False`) and `scrip_grid_file` (`optional(str)`, default `None`) to `veros/settings.py`, following the existing `Setting(default, type, description)` pattern. This flag selects which grid-init routine runs, and (Phase 6) whether the momentum metric term is computed at all — not array shape, which is unconditional.

## Phase 1 — Promote grid-metric Variables to 2D + fix the netCDF collision

Depends on: Phase 0. Blocks everything else.

- `veros/variables.py`: change `dims` for `xt, xu, yt, yu, dxt, dxu, dyt, dyu, cost, cosu, tantr` from 1D (`XT/XU/YT/YU`) to `T_HOR/U_HOR/V_HOR/ZETA_HOR` as appropriate, mirroring `coriolis_t`/`kbot`/`area_*`. No new variables, no renames.
- `veros/io_tools/netcdf.py`: split `initialize_file` into two passes — register every netCDF dimension first (sized from `state.dimensions[dim]` directly, not derived from a co-named variable's reversed `dims` tuple), then create variables (including now-2D `xt`/`yt`/`xu`/`yu`) in a second pass once all dimensions they need already exist. Fixes the two confirmed bugs (dimension mis-sizing at the old line 79, premature variable creation at the old line 98) without any renaming — verified via direct `h5netcdf`/`xarray` testing that a variable sharing a name with one of its own (now two) dimensions writes and reads back cleanly.
- `veros/diagnostics/snapshot.py`: no change needed — `xt/xu/yt/yu` stay in `DEFAULT_OUTPUT_VARS` under their existing names.
- Land together with Phase 3 (same PR sequence) — Phase 1 alone leaves `numerics.py` writing 1D-shaped values into now-2D slots, which won't run.
- Verification: import-level check that `VARIABLES["dxt"].dims == T_HOR`; real run-verification deferred to Phase 3's gate.

## Phase 2 — SCRIP grid reader, built on the existing `earthsystemgrids` library (parallelizable with Phase 1)

Depends on: nothing.

**Reuse, don't reinvent.** `DisplacedPoleGrid.SCRIP.nc` was generated by `/home/tienyiao/projects_local/EarthSystemGrids.py` (`earthsystemgrids` on PyPI-style packaging, same author as this project). That library already has exactly the machinery this phase needs, confirmed by reading it:

- `EarthSystemGrids.base.UnstructuredGridMesh` / `StructuredQuadMesh` (`base/UnstructuredGridMesh.py`, `base/StructuredQuadMesh.py`) is a full node/edge/face mesh object: `node_lon/node_lat` (unique corners), `face_lon/face_lat` (cell centres), `face_nodes`/`face_edges`/`edge_nodes` (structured (nj,ni) connectivity, built by `_build_topology_2d`), `edge_length` (great-circle length in metres, **already computed** — this is exactly Veros's `dxt`/`dyt`), `area` (m²), `mask`.
- `StructuredQuadMesh.rotation_angle()` computes the grid-i-direction-to-true-east angle via exact spherical geometry (unit-vector cross/dot products at each face's corners, not a flat-Earth `cos(lat)` finite difference) — and `write_to_SCRIP_grid_file` (`StructuredQuadMesh.py:70-136`) uses exactly this method to produce the `grid_angle`/`grid_cos_angle`/`grid_sin_angle` fields already sitting in `DisplacedPoleGrid.SCRIP.nc`. Reading the file's angle fields directly, or reconstructing via `rotation_angle()` on the read-back mesh, are two independent computations of the same quantity — comparing them is a free correctness check.
- `StructuredQuadMesh.from_corners(corner_lon, corner_lat, face_lon, face_lat, area, mask, shape)` (`StructuredQuadMesh.py:26-68`) builds the full mesh from exactly the arrays a SCRIP file provides (`grid_corner_lat/lon`, `grid_center_lat/lon`, `grid_area`, `grid_imask`), modulo unit conversion (SCRIP: degrees and radians²; the library: radians and m², via `_R_EARTH = 6.371e6` at `UnstructuredGridMesh.py:7`) and axis order (SCRIP `grid_dims=[nx,ny]` vs. the library's `shape=(nj,ni)`).
- **Done, not a gap**: `StructuredQuadMesh.from_SCRIP_file(path)` now exists in the library (`EarthSystemGrids/base/StructuredQuadMesh.py:139-203`), the direct inverse of `write_to_SCRIP_grid_file` — reads `grid_dims`/`grid_center_lat/lon`/`grid_corner_lat/lon`/`grid_area`/`grid_imask`, handles unit conversion (degrees→radians via each variable's own `units` attribute, radians²→m² via `_R_EARTH`) and the `grid_dims=[ni,nj]`/`shape=(nj,ni)` axis convention, and calls `from_corners(...)`. **Verified working against the real `DisplacedPoleGrid.SCRIP.nc`**: shape comes out `(59,120)` matching `grid_dims`; `sum(area)` = 5.0985e14 m² vs. Earth's true surface area 5.1006e14 m² (0.04% agreement); edge lengths mostly fall in the physically expected 10-1000 km range for this grid's resolution. One modeling note for Phase 4, not a reader bug: exactly `nx`=120 edges come out near-zero (~1e-11 m), consistent with `DisplacedPoleGrid`'s documented pole-closing behavior (`ring_radius=0` collapses the last ring to a point) — but `grid_imask` marks that ring as ocean, not land, so Veros will need to either mask it explicitly or confirm the production grid uses `ring_radius>0`, since a near-zero `dxt`/`dyt` there would blow up CFL/division. Flagged for Phase 4's verification step, not blocking Phase 2.

**Remaining work (Veros side only):**
1. In Veros, add a thin adapter `veros/tools/scrip.py` (exported via `veros/tools/__init__.py`) that calls `earthsystemgrids.StructuredQuadMesh.from_SCRIP_file(...)` and derives Veros's Arakawa-C-staggered 2D arrays from the returned mesh's existing structured connectivity — no new spherical-geometry math required:
   - T-point `xt`/`yt` = `face_lon`/`face_lat` reshaped to `(nx,ny)`.
   - `dxt`/`dyt` (T-cell width/height) = `edge_length` at the cell's horizontal/vertical bounding edges (`face_edges[:,0]`/`[:,2]` for the i-direction edges, `[:,1]`/`[:,3]` for the j-direction edges, per `_build_topology_2d`'s documented edge ordering).
   - U-point `xu`/`yu` = midpoint of the shared vertical edge between T(i,j) and T(i+1,j) (i.e. `node_lon/node_lat` at `face_nodes[:,1]`/`face_nodes[:,2]`, the SE/NE corners); `dxu` = great-circle distance between adjacent `face_lon/face_lat` centres. V/Z-point analogues follow the same pattern.
   - `grid_angle`/`cos_angle`/`sin_angle` are present in the file but **not consumed** — no rotation is applied anywhere (forcing is assumed grid-aligned, an explicit scope decision — see Context section above, not Phase 0).
   - U-point placement (`face_nodes[:,1]`/`[:,2]`, the SE/NE shared-edge midpoint) is this adapter's own engineering choice about which side of T(i,j) the U-point sits on, based on the SW/SE/NE/NW corner-ordering convention `_build_topology_2d` documents — it has **not** been independently checked against Veros's own existing C-grid staggering convention (i.e., which physical side `maskU`/`dxu` etc. are defined on today). Verify this alignment explicitly during implementation, ideally via the Phase 3/Phase 4 fp-tolerance diff on the regular-grid fixture, which would catch a systematic off-by-one-half-cell error here.
   - Applies the same ghost-cell/`enable_cyclic_x` padding conventions as the legacy `calc_grid_spacings_kernel`.
   - Depends on `settings.radius` for any unit reconciliation, not the library's own hardcoded `_R_EARTH` — flag if they differ (both are 6.371e6 today, so this is a latent, not active, discrepancy).
2. Also add `regular_grid_to_scrip()` (in either the library or `veros/tools/scrip.py`, wherever the corner-generation logic naturally lives) — builds a SCRIP-equivalent description of the *existing* `global_4deg` regular grid (nx=90, ny=40, dx=dy=4°). **This is the fixture Phase 4 and Phase 8 both depend on** for their fp-tolerance regression gates (see the Standing Invariant above — the other phases use the existing pyOM-consistency/setup suites instead).
- Verification: new `test/scrip_reader_test.py` in Veros — sanity-check shapes/ranges on `DisplacedPoleGrid.SCRIP.nc` via the new adapter; confirm the regular-grid-as-SCRIP fixture, round-tripped through `from_SCRIP_file` + the Veros adapter, matches `numerics.calc_grid_spacings_kernel`/`calc_grid_metrics_kernel`'s current output to fp tolerance. `from_SCRIP_file` already has its own test coverage in the `earthsystemgrids` repo (`tests/base/test_StructuredQuadMesh.py`) — that's that repo's responsibility to maintain, not Veros's.

## Phase 3 — Rework the legacy analytic path onto the 2D arrays (regression baseline)

Depends on: Phase 1. Blocks Phase 4/5/6 gates and blocks `main` being green after Phase 1.

- `veros/core/numerics.py::calc_grid_spacings_kernel`: keep `u_centered_grid`'s 1D profile logic, but broadcast the resulting profile into the (now 2D) `xt/yt/xu/yu/dxt/dyt/dxu/dyu` slots (`xt[i,j] = xt_1d[i]`, `yt[i,j] = yt_1d[j]`) — reproduces today's values exactly, by construction.
- `calc_grid_metrics_kernel`: `cost/cosu/tantr/area_*` become elementwise on 2D `yt/yu` — formulas unchanged, only shape changes.
- `calc_beta`: generalize `(coriolis_t[:,3:-1] - coriolis_t[:,2:-2]) / dyu[2:-2]` to fully 2D-indexed (`dyu[2:-2,2:-2]`, matching `coriolis_t`'s slice indices) — reduces exactly to the current formula on a legacy grid since `dyu` is constant along `i` there.
- `veros/veros.py`: update `VerosSetup.set_grid`/`set_coriolis` abstract-method docstrings (~lines 78-110) for the (nx,ny)-shaped contract; note that setups assigning a literal 1D array to `vs.dxt` (per the old docstring example) need an explicit broadcast (`vs.dxt = 4.0 * npx.ones_like(vs.dxt)`-style code, already used in the shipped setups, continues to work unchanged).
- Verification: `test/pyom_consistency/4deg_test.py`, `numerics_test.py`, and `test/setup_test.py` (all setups) must pass unchanged — these are the existing strongest regression net. Add explicit shape/broadcast-correctness assertions in a new `test/curvilinear_grid_test.py`.

## Phase 4 — New curvilinear grid-init path (`calc_grid_scrip`)

Depends on: Phases 1, 2, 3.

- `veros/core/numerics.py`: add `calc_grid_scrip(state)`, reading `settings.scrip_grid_file` and populating `xt/yt/xu/yu/dxt/dyt/dxu/dyu/cost/cosu/area_*` from the Phase-2 adapter's output, applying the same ghost-cell/`enable_cyclic_x` padding as the legacy path. Does **not** compute `tantr` — per Phase 6, nothing on the curvilinear path consumes it, and producing a value for it now would require the exact general-metric-term derivation that Phase 6 explicitly defers. It's left at its allocation default (unused, not a correctness gap given Phase 6's gating).
- **Open design question, flagged not resolved:** whether `cost`/`cosu` become `cos(lat)` (as today) or `1.0` everywhere on the curvilinear path once `dxt`/`dyt` already carry true physical along-i/along-j distances from SCRIP corner geometry — getting this wrong double-counts or drops a metric factor in `area_t`, friction's `cost**hor_friction_cosPower` scaling, and the elliptic solver's `cost`/`cosu` terms. Resolve by cross-checking against MOM4/POP's generalized-orthogonal-coordinate formulation (where `dxt`/`dyt` are literal arc lengths with no separate `cos(lat)` factor) before writing this code — a short written note, not guesswork.
- Single `calc_grid` entry point, branching internally on `settings.enable_curvilinear_grid` between `calc_grid_scrip` and the existing analytic path — keeps `VerosSetup.setup()`'s call sequence in `veros/veros.py` untouched.
- `set_coriolis` needs no core change — `coriolis_t = 2*omega*sin(yt*pi/180)` already works once `yt` is real 2D geographic latitude (confirmed the only place outside `numerics.py` reading `vs.yt` directly).
- Verification: diff every array `calc_grid_scrip` actually produces (`xt,yt,xu,yu,dxt,dyt,dxu,dyu,cost,cosu,area_t,area_u,area_v` — **not** `tantr`, which this phase deliberately doesn't compute) between `calc_grid_scrip` and the legacy path on the Phase-2 regular-grid-as-SCRIP fixture — must match to fp tolerance. Dedicated test in `test/curvilinear_grid_test.py`.

## Phase 5 — Mechanical 1D→2D substitution across consumer kernels

Depends on: Phase 1 **and Phase 3** — this is a hard prerequisite, not merely a benefit: Phase 1 alone leaves `numerics.py` crashing at grid init (writes 1D data into 2D slots), and Phase 5's own verification (`pyom_consistency`/`setup_test`/`linear_solver_test`) requires a working end-to-end model to run at all, which only exists once Phase 3 lands. Independent of Phase 4 and Phase 6.

Confirmed-mechanical files (broadcast-pattern substitution only, no algorithm change): `friction.py`, `isoneutral/isoneutral.py`, `isoneutral/diffusion.py`, `advection.py`, `diffusion.py`, `eke.py`, `idemix.py`, `tke.py`, `thermodynamics.py`, `external/poisson_matrix.py`, `external/solve_stream.py`, `external/streamfunction_init.py`, `external/solve_pressure.py`, `external/line_integrals.py`, `diagnostics/cfl_monitor.py`. Also includes `momentum.py`'s non-metric Coriolis-advection terms (the `dxt/dxu/dyt/dyu/cost/cosu` scaling in `tend_coriolisf`'s main block) — mechanical like everything else here. Excludes only the `tantr` metric-term block itself (Phase 6). Solver backends (`solvers/scipy.py`, `scipy_jax.py`, `petsc_.py`) need zero changes.

Land as a small number of independently-testable commits grouped by subsystem (e.g. friction+isoneutral+advection; elliptic-solver stack; tke/eke/idemix/thermodynamics), each passing the full regression suite before the next.

Verification: `test/pyom_consistency/*_test.py`, `test/setup_test.py`, `test/linear_solver_test.py` — zero output difference on all legacy setups.

## Phase 6 — Explicitly disable the momentum metric term on the curvilinear path (deferred, not derived)

Depends on: Phase 5. Practical, not conceptual: Phase 5's mechanical rewrite of `momentum.py::tend_coriolisf`'s non-metric terms and this phase's gating of the same function's metric-term block touch the same code region, so land this one second to avoid a merge conflict. (Also needs Phase 0's `enable_curvilinear_grid` setting to exist, which Phase 5 doesn't otherwise require — already satisfied by that point.)

**Out of scope by explicit decision**: deriving the general orthogonal-curvilinear form of the metric term. Judged high enough physics risk to be its own separate engineering task, not bundled into this grid-generalization project. What this phase actually does is much smaller — make sure the curvilinear path doesn't silently run with a formula that's known to be wrong:

- `veros/core/momentum.py::tend_coriolisf`: gate the existing `tantr`-based `du_cor`/`dv_cor` block behind `if settings.coord_degree and not settings.enable_curvilinear_grid:` (or equivalent) — i.e. comment out / skip its execution when `enable_curvilinear_grid=True`, while leaving the block's code physically in place (commented, not deleted) as the starting point for the follow-on task. Add a clear comment at the site stating why it's disabled and pointing at the follow-on task, not just a bare `# TODO`.
- On `enable_curvilinear_grid=True`, log a one-time startup warning (matching the style of other setting-driven warnings in the codebase) noting the momentum equation is running without the grid-curvature metric term, so this is a visible, documented limitation of the initial curvilinear delivery rather than a silent gap.
- The legacy analytic path (`enable_curvilinear_grid=False`) is completely untouched — `tantr` keeps computing and running exactly as today, byte-for-byte, so `test/pyom_consistency/momentum_test.py`/`4deg_test.py` need no changes and keep passing as the regression net they already are.
- Verification: confirm the legacy-path regression tests are unaffected (the only thing that could break them is touching code they don't reach); confirm the curvilinear path runs without the metric-term block (no `tantr`-shaped array involved, no crash on the now-2D grid) and emits the startup warning.

This phase removes the single highest-risk item from this plan's critical path. The general-formula derivation is a follow-on task, not blocked by anything else here — it can start any time once Phase 1's 2D arrays exist, independent of Phase 8.

## Phase 8 — Migrate the test case to a real curvilinear setup

Depends on: Phases 4, 5, 6.

- `test_case_displaced_pole_grid/global_4deg.py`: set `enable_curvilinear_grid=True`, `scrip_grid_file=...`, `nx,ny=120,59`; `set_grid` no longer sets `dxt/dyt` analytically (Phase 4 derives them); `set_topography`/`set_initial_conditions`/`set_forcing` regrid existing forcing/bathymetry data onto the new grid. Vector forcing fields (`taux`/`tauy`) are applied as-is, assumed already grid-aligned — no rotation step (explicit scope decision, not deferred work). The momentum equation runs without the metric term (Phase 6) — expect and accept the startup warning; this is the known, documented limitation of this delivery, not a bug to chase.
- **Scope recommendation for first deliverable**: rather than build a general runtime regridder, produce a one-time offline-regridded asset (`DisplacedPoleGrid_forcing.nc`) using the Phase-2 SCRIP machinery, and ship it as a new asset entry — defers the general regridding problem. Flagging this as a scope decision worth confirming, not unilaterally deciding.
- Verification:
  1. **Regular-grid-clone integration test**: a parametrized variant of the test setup pointed at the regular-grid-as-SCRIP fixture, run end-to-end (`setup()`+`run()`, a few timesteps), diffed against the untouched legacy `GlobalFourDegreeSetup` — the strongest gate in the whole plan.
  2. **Real displaced-pole smoke test**: instantiate the real setup, run a few steps, assert no NaN/Inf (reuse `numerics.sanity_check`), assert basic physical plausibility (bounded depths, sane mask fractions, T/S in plausible ranges).

## Phase 9 — Restart polish (optional)

Depends on: Phase 4 (conceptually — nothing to polish before the curvilinear grid-init path exists). Not blocking; can be picked up any time after.

Grid metrics already have `write_to_restart=False` and are cheaply recomputed on startup — fine as-is for the curvilinear path too (SCRIP read is deterministic and cheap). No change required unless it becomes a startup bottleneck later.

## Sequencing

Shown as independent branches rather than one nested tree, since Phase 2 has no prerequisites at all and Phase 8 needs two separate branches to converge (each phase's own "Depends on" line above is the authoritative statement; this is just a visual summary):

```
Phase 0 → Phase 1 → Phase 3                    (2D promotion, then the legacy path must work end-to-end again)
Phase 2                                        (no dependencies — runs any time)

Phase 3 ─┬─→ Phase 4 ←── also needs Phase 2
         └─→ Phase 5 → Phase 6

Phase 4 ──┬─→ Phase 8                          (needs BOTH Phase 4 and Phase 6)
Phase 6 ──┘
Phase 4 ───→ Phase 9 (optional — independent of Phase 6 and Phase 8)
```

**Phase 3, not Phase 1, is the real fork point** — Phase 1 alone leaves `numerics.py` crashing at grid init, so nothing downstream can actually run (and be regression-tested) until Phase 3 lands. From there, two branches run in parallel: (a) `Phase 4` — the curvilinear grid-array substrate, which also needs Phase 2 (the SCRIP adapter) directly; (b) `Phase 5 → Phase 6` — the consumer-kernel rewrite, with Phase 6 sequenced after Phase 5 only because they touch the same function (`tend_coriolisf`), not a real data dependency. Phase 2 has no dependencies and can start immediately, in parallel with everything else, until Phase 4 needs its output. Phase 8 is the integration point and needs both branches (4 and 6) done; Phase 9 only needs Phase 4 and can proceed independently of both 6 and 8. The general metric-term derivation itself is a separate follow-on task, not on this critical path at all.

## Critical files

- `veros/variables.py` — Variable-dims promotion (Phase 1), highest blast radius
- `veros/core/numerics.py` — legacy path rework (Phase 3) + new `calc_grid_scrip` (Phase 4)
- `veros/core/momentum.py` — metric term gated off on the curvilinear path, commented not deleted (Phase 6); the actual physics-derivation risk is explicitly deferred out of this plan
- `veros/io_tools/netcdf.py` — dimension-registration/sizing fix, required the moment `xt`/`yt` go 2D (Phase 1)
- `veros/tools/scrip.py` (new) — thin adapter over `earthsystemgrids.StructuredQuadMesh.from_SCRIP_file` + regular-grid-as-SCRIP fixture generator that Phase 4's and Phase 8's regression gates depend on (Phase 2)
- `test_case_displaced_pole_grid/global_4deg.py` — integration target (Phase 8)

## Verification summary

Two kinds of verification run through this plan, not one: the SCRIP-fixture fp-tolerance check (Phase 4, per-array; Phase 8, end-to-end) wherever a phase introduces a new SCRIP-consuming path, and the existing pyOM-consistency/setup regression suites (`test/pyom_consistency/*`, `test/setup_test.py`, `test/linear_solver_test.py`) wherever a phase only touches the legacy analytic path (3, 5, 6) — those suites are the only ground truth against reference pyOM output available today, and every phase in this plan is designed to leave them green throughout, including Phase 6's gating change. New tests live in `test/curvilinear_grid_test.py` and `test/scrip_reader_test.py`.
