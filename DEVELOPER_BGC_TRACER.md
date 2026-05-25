# BGC Tracer Developer Guide

This document describes the BGC (biogeochemical) passive tracer framework added to Veros, inspired by [veros-bgc](https://github.com/team-ocean/veros-bgc).

## Overview

The framework transports an arbitrary number of passive tracers through advection, Adams-Bashforth time stepping, implicit vertical mixing, and surface forcing. Tracers are registered by name and handled generically — adding a new tracer requires no changes to core code.

## Files Changed

| File | Change |
|---|---|
| `veros/settings.py` | Added `bgc_tracer_names` setting |
| `veros/core/bgc_tracers.py` | New file: `register_bgc_tracer` helper, generic transport kernels, and main routine |
| `veros/veros.py` | Call `integrate_bgc_tracers` in the main time-stepping loop |

## How It Works

### Registration

Use `register_bgc_tracer` in `set_parameter` to register a tracer. It adds the name to `bgc_tracer_names` and declares the three required transport variables in `var_meta` atomically:

```python
from veros.core.bgc_tracers import integrate_bgc_tracers, register_bgc_tracer

register_bgc_tracer(state, "dic", units="mol/m^3", long_name="Dissolved inorganic carbon")
register_bgc_tracer(state, "alk", units="mol/eq/m^3", long_name="Alkalinity")
```

Any setup-specific variables (e.g. relaxation targets, restoring time scales) are registered separately in `var_meta`:

```python
var_meta = state.var_meta
var_meta.update(
    dic_star=Variable("dic_star", ("yt",), "mol/m^3", "Reference surface DIC"),
    dic_rest=Variable("dic_rest", ("xt", "yt"), "1/s", "DIC restoring time scale"),
    # ...
)
```

### Variable Naming Convention

`register_bgc_tracer` registers exactly these three variables per tracer:

| Variable | Dims | Purpose |
|---|---|---|
| `{name}` | `("xt", "yt", "zt", "timesteps")` | Tracer concentration |
| `d_{name}` | `("xt", "yt", "zt", "timesteps")` | Adams-Bashforth tendency storage |
| `forc_{name}_surface` | `("xt", "yt")` | Surface forcing (set each timestep in `set_forcing`) |

All other variables needed for a specific forcing scheme (e.g. `{name}_star`, `{name}_rest`) are the user's responsibility.

### Transport Pipeline

Each timestep, `veros/veros.py` calls `integrate_bgc_tracers(state)` immediately after `thermodynamics.thermodynamics(state)`. For each tracer in `bgc_tracer_names`, three steps run in order:

1. **`_advect_bgc_tracer`** — computes the 3D advection tendency into `d_{name}[..., tau]` using the same `advect_tracer` kernel as temperature and salinity.

2. **`_ab_step_bgc_tracer`** — advances the tracer to `taup1` using Adams-Bashforth:
   ```
   tracer[taup1] = tracer[tau] + dt * ((1.5 + AB_eps)*d[tau] - (0.5 + AB_eps)*d[taum1]) * maskT
   ```

3. **`_vertmix_bgc_tracer`** — solves the implicit vertical diffusion tridiagonal system (same `kappaH` as temperature/salinity) and applies `forc_{name}_surface`. Enforces cyclic boundary conditions after the solve.

The transport loop is skipped entirely when `bgc_tracer_names` is empty (zero overhead for runs without BGC tracers).

### Setting Initial Conditions and Forcing

In `set_initial_conditions`, initialise the tracer and any setup-specific variables:

```python
vs.dic = update(vs.dic, at[...], 0.0)
vs.dic_star = npx.where((vs.yt > 10) & (vs.yt < 15), 2200.0, 2000.0)
vs.dic_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]
```

In `set_forcing`, compute `forc_{name}_surface` each timestep:

```python
vs.forc_dic_surface = vs.dic_rest * (vs.dic_star - vs.dic[:, :, -1, vs.tau])
```

### Adding a Second Tracer

1. Call `register_bgc_tracer(state, "alk", ...)` in `set_parameter`.
2. Add any setup-specific variables to `var_meta` manually.
3. Set initial conditions and forcing in `set_initial_conditions` / `set_forcing`.

No changes to `bgc_tracers.py` or `veros.py` needed.

## What Is Not Included

- Horizontal (isoneutral/skew) diffusion for BGC tracers — the `isoneutral_diffusion_kernel` in Veros hardcodes `dtemp_iso`/`dsalt_iso` outputs and cannot be reused generically without refactoring.
- Tracer-tracer interactions (sources/sinks) — implement these in `set_forcing` or `after_timestep` directly.
