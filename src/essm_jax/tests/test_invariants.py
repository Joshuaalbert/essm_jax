"""Numerical regressions for the contracts documented in INVARIANTS.md."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from essm_jax.essm import (
    ExtendedStateSpaceModel,
    convert_filter_state_to_incremental_state,
)

tfpd = tfp.distributions


def _linear_model(materialise_jacobians=False):
    def transition(z, t, t_next, args):
        drift, = args
        return tfpd.MultivariateNormalTriL(
            0.8 * z + drift * (t_next - t), jnp.array([[0.4]])
        )

    def observation(z, t, args):
        return tfpd.MultivariateNormalTriL(1.2 * z, jnp.array([[0.7]]))

    return ExtendedStateSpaceModel(
        transition_fn=transition,
        observation_fn=observation,
        initial_state_prior=tfpd.MultivariateNormalTriL(
            jnp.array([0.3]), jnp.array([[1.1]])
        ),
        materialise_jacobians=materialise_jacobians,
        dt=0.25,
    )


@pytest.mark.parametrize("materialise_jacobians", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_filter_matches_scalar_kalman_reference(materialise_jacobians, missing):
    model = _linear_model(materialise_jacobians)
    observations = np.array([0.1, -0.2, 0.4, 0.7])
    masked = np.array([False, missing, False, False])
    mask = jnp.asarray(masked) if missing else None
    drift = 0.2

    # Independent scalar Kalman equations; the first observation follows a transition.
    mean, variance, log_prob = 0.3, 1.1 ** 2, 0.0
    means, variances, log_probs = [], [], []
    for value, is_missing in zip(observations, masked):
        mean = 0.8 * mean + drift * model.dt
        variance = 0.8 ** 2 * variance + 0.4 ** 2
        if not is_missing:
            residual = value - 1.2 * mean
            innovation_variance = 1.2 ** 2 * variance + 0.7 ** 2
            mean += variance * 1.2 / innovation_variance * residual
            variance *= 0.7 ** 2 / innovation_variance
            log_prob -= 0.5 * (
                np.log(2 * np.pi * innovation_variance) + residual ** 2 / innovation_variance
            )
        means.append(mean)
        variances.append(variance)
        log_probs.append(log_prob)

    observations = jnp.asarray(observations[:, None])  # [num_time, observation_size]
    result = jax.jit(
        lambda values: model.forward_filter(values, transition_args=(drift,), mask=mask, t0=2.0)
    )(observations)
    np.testing.assert_allclose(result.t, 2.0 + np.arange(1, 5) * model.dt)
    np.testing.assert_allclose(result.filtered_mean[:, 0], means, rtol=1e-6)
    np.testing.assert_allclose(result.filtered_cov[:, 0, 0], variances, rtol=1e-6)
    np.testing.assert_allclose(result.log_cumulative_marginal_likelihood, log_probs, rtol=1e-6)
    np.testing.assert_allclose(model.log_prob(observations, transition_args=(drift,), mask=mask, t0=2.0), log_prob)

    smoothed, prior = model.backward_smooth(result, transition_args=(drift,), include_prior=True)
    np.testing.assert_array_equal(smoothed.t, result.t)
    np.testing.assert_allclose(smoothed.smoothed_mean[-1], result.filtered_mean[-1])
    np.testing.assert_allclose(smoothed.smoothed_cov[-1], result.filtered_cov[-1])
    assert np.all(np.isfinite(prior.mean))
    assert np.all(prior.covariance >= 0)
    posterior = model.posterior_marginals(observations, transition_args=(drift,), mask=mask, t0=2.0)
    np.testing.assert_allclose(posterior.smoothed_mean, smoothed.smoothed_mean)
    np.testing.assert_allclose(posterior.smoothed_cov, smoothed.smoothed_cov)


def test_missing_update_preserves_prediction_at_explicit_time():
    model = _linear_model()
    initial = model.create_initial_filter_state(t0=2.0)
    predicted = model.incremental_predict(initial, transition_args=(0.2,), t_next=jnp.array(2.75))
    np.testing.assert_allclose(predicted.t, 2.75)
    np.testing.assert_allclose(predicted.filtered_mean, [0.8 * 0.3 + 0.2 * 0.75])
    np.testing.assert_allclose(predicted.filtered_cov, [[0.8 ** 2 * 1.1 ** 2 + 0.4 ** 2]])

    updated, _ = model.incremental_update(
        predicted, jnp.array([np.nan]), mask=jnp.array(True)
    )
    for actual, expected in zip(updated, predicted):
        np.testing.assert_array_equal(actual, expected)


def test_forward_simulation_agrees_for_batch_and_incremental_state():
    model = _linear_model()
    filtered = model.forward_filter(jnp.zeros((3, 1)), transition_args=(0.2,), t0=2.0)
    terminal = convert_filter_state_to_incremental_state(filtered)
    np.testing.assert_array_equal(terminal.filtered_mean, filtered.filtered_mean[-1])
    np.testing.assert_array_equal(terminal.filtered_cov, filtered.filtered_cov[-1])
    np.testing.assert_array_equal(
        terminal.log_cumulative_marginal_likelihood,
        filtered.log_cumulative_marginal_likelihood[-1],
    )
    key = jax.random.PRNGKey(17)
    batch_samples = model.forward_simulate(key, 3, filtered, transition_args=(0.2,))
    incremental_samples = model.forward_simulate(key, 3, terminal, transition_args=(0.2,))
    for batch, incremental in zip(batch_samples, incremental_samples):
        np.testing.assert_array_equal(batch, incremental)
    np.testing.assert_allclose(batch_samples.t, terminal.t + np.arange(1, 4) * model.dt)


@pytest.mark.parametrize(
    "observations, mask, message",
    [
        (jnp.zeros(3), None, "observations must be a 2D array"),
        (jnp.zeros((3, 1)), jnp.zeros((3, 1), dtype=bool), "mask must be a 1D array"),
        (jnp.zeros((3, 1)), jnp.zeros(2, dtype=bool), "mask and observations must have the same length"),
    ],
)
def test_filter_rejects_invalid_observation_and_mask_shapes(observations, mask, message):
    with pytest.raises(ValueError, match=message):
        _linear_model().forward_filter(observations, transition_args=(0.2,), mask=mask)
