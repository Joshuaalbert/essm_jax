import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from essm_jax.dual_essm import DualExtendedStateSpaceModel
from essm_jax.essm import ExtendedStateSpaceModel

tfpd = tfp.distributions


@pytest.mark.parametrize("masked", [False, True])
def test_dual_filter_coupling_and_mask(masked):
    def transition(value, t, t_next, other):
        return tfpd.MultivariateNormalTriL(
            value + other * (t_next - t), jnp.array([[0.2]])
        )

    def observation(value, t, other):
        return tfpd.MultivariateNormalTriL(value * other, jnp.array([[0.5]]))

    models = [
        ExtendedStateSpaceModel(
            transition_fn=transition,
            observation_fn=observation,
            initial_state_prior=tfpd.MultivariateNormalTriL(jnp.array([mean]), jnp.ones((1, 1))),
            dt=0.25,
        )
        for mean in (2.0, 3.0)
    ]
    dual = DualExtendedStateSpaceModel(*models)
    initial = dual.create_initial_filter_state(t0=1.0)
    predicted = dual.incremental_predict(initial)
    # Both predictions use the input estimates, not the other's new prediction.
    np.testing.assert_allclose(predicted.state_filter_state.filtered_mean, [2.0 + 3.0 * 0.25])
    np.testing.assert_allclose(predicted.param_filter_state.filtered_mean, [3.0 + 2.0 * 0.25])
    for state in predicted:
        np.testing.assert_allclose(state.t, 1.25)
        np.testing.assert_allclose(state.filtered_cov, [[1.04]])
        np.testing.assert_allclose(state.log_cumulative_marginal_likelihood, 0.0)

    updated, marginal = dual.incremental_update(initial, jnp.array([7.0]), mask=jnp.array(masked))
    np.testing.assert_allclose(marginal.mean(), [6.0])
    np.testing.assert_allclose(marginal.covariance(), [[9.25]])
    for state in updated:
        np.testing.assert_allclose(state.t, 1.0)

    if masked:
        for actual_state, initial_state in zip(updated, initial):
            for actual, expected in zip(actual_state, initial_state):
                np.testing.assert_array_equal(actual, expected)
    else:
        # Scalar Kalman gains expose whether parameter inference uses the new state.
        state_mean = 2.0 + 3.0 / 9.25 * (7.0 - 6.0)
        param_innovation_variance = state_mean ** 2 + 0.25
        param_mean = 3.0 + state_mean / param_innovation_variance * (7.0 - 3.0 * state_mean)
        np.testing.assert_allclose(updated.state_filter_state.filtered_mean, [state_mean])
        np.testing.assert_allclose(updated.state_filter_state.filtered_cov, [[0.25 / 9.25]])
        np.testing.assert_allclose(updated.param_filter_state.filtered_mean, [param_mean])
        np.testing.assert_allclose(
            updated.param_filter_state.filtered_cov, [[0.25 / param_innovation_variance]]
        )
