# do not import veros.core here at module level -- see note in veros.py
import jax
import jax.numpy as jnp


def detect_nan_in_state(state, header=""):
    """Scan all active variables in ``state`` for NaN / Inf values.

    Meant to be sprinkled between sub-steps of :meth:`VerosSetup.step` to narrow
    down which routine first introduces non-finite values.

    Reporting is done via jax.debug.print, so it works whether this is called
    eagerly or while being traced (jax.jit/grad/jvp).

    Controlled by the ``enable_nan_checks`` setting; a no-op (and adds nothing
    to the traced jaxpr) when that setting is off.

    Arguments:
        state: VerosState instance to inspect.
        header (str): Label identifying the calling site, included in the report.
    """
    if not state.settings.enable_nan_checks:
        return

    vs = state.variables

    for name in vs.fields():
        val = vs.get(name)

        if val is None or not hasattr(val, "dtype"):
            print(f"Skip nan check of variable {name} because it is either None, or does not have dtype.")
            continue

        if not jnp.issubdtype(val.dtype, jnp.floating):
            print(f"Skip nan check of variable {name} because its type is not float.")
            continue

        n_nan = jnp.sum(jnp.isnan(val))
        n_inf = jnp.sum(jnp.isinf(val))
        has_bad = (n_nan != 0) | (n_inf != 0)

        message = f"Non-finite values detected at '{header}': {name}: " "{n_nan} NaN, {n_inf} Inf"

        def _report(n_nan, n_inf, message=message):
            jax.debug.print(message, n_nan=n_nan, n_inf=n_inf)
            return 0

        def _noop(n_nan, n_inf):
            return 0

        jax.lax.cond(has_bad, _report, _noop, n_nan, n_inf)
