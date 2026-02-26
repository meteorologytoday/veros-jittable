import pytest
def test_jittable_veros():

    import jax
    from veros import runtime_settings

    setattr(runtime_settings, "backend", "jax")
    setattr(runtime_settings, "force_overwrite", True)
    setattr(runtime_settings, 'linear_solver', 'scipy_jax')
    setattr(runtime_settings, 'device', 'cpu')

    from veros.setups.acc import ACCSetup

    veros_object = ACCSetup()
    veros_object.setup()

    @jax.jit
    def step_function(state, step):
        """
            Convert the state function into a "pure step" copying the input state
        """
        n_state = state
        veros_object.step(n_state)  # This is a function that modifies state object inplace
        return n_state

    warmup_steps = 2
    state = veros_object.state 

    for step in range(warmup_steps):
        state = step_function(state, step)

