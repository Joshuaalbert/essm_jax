import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from essm_jax.essm import ExtendedStateSpaceModel

tfd = tfp.distributions


def linear_model(diagonal=True, dense=False, dtype=jnp.float64):
    transition = jnp.array([[0.8, 0.15], [-0.1, 0.7]], dtype=dtype)
    projection = jnp.array([[1., 0.2], [-0.3, 0.7], [0.5, 1.]], dtype=dtype)

    def normal(mean, std):
        std = jnp.asarray(std, dtype=dtype)
        if diagonal:
            return tfd.MultivariateNormalDiag(mean, std)
        return tfd.MultivariateNormalTriL(mean, jnp.diag(std))

    return ExtendedStateSpaceModel(
        transition_fn=lambda z, t, t_next, args: normal(transition @ z, args[0] * jnp.array([0.3, 0.6], dtype=dtype)),
        observation_fn=lambda z, t, args: normal(projection @ z, [0.2, 0.7, 1.3]),
        initial_state_prior=normal(jnp.array([0.1, -0.2], dtype=dtype), [0.8, 1.2]),
        materialise_jacobians=dense,
    )


@pytest.mark.parametrize("dense", [False, True])
def test_diagonal_scales_match_covariances_and_likelihood_gradients(dense):
    observations = jnp.array([[0.2, -0.1, 0.3], [0.7, 0.4, -0.5], [0.1, 0.2, 0.6]])
    diag = linear_model(diagonal=True, dense=dense)
    tril = linear_model(diagonal=False, dense=dense)
    diag_filter = jax.jit(diag.forward_filter)(observations, transition_args=(1.,))
    tril_filter = jax.jit(tril.forward_filter)(observations, transition_args=(1.,))
    for actual, expected in zip(diag_filter, tril_filter):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    for actual, expected in zip(diag.backward_smooth(diag_filter, transition_args=(1.,)),
                                tril.backward_smooth(tril_filter, transition_args=(1.,))):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    for model in (diag, tril):
        np.testing.assert_allclose(model.log_prob(observations, transition_args=(1.,)),
                                    tril_filter.log_cumulative_marginal_likelihood[-1])
    diag_grad = jax.jit(jax.grad(lambda scale: diag.log_prob(observations, transition_args=(scale,))))(1.)
    tril_grad = jax.jit(jax.grad(lambda scale: tril.log_prob(observations, transition_args=(scale,))))(1.)
    np.testing.assert_allclose(diag_grad, tril_grad, rtol=1e-10, atol=1e-10)
    step = 1e-5
    finite_difference = (diag.log_prob(observations, transition_args=(1. + step,))
                         - diag.log_prob(observations, transition_args=(1. - step,))) / (2 * step)
    np.testing.assert_allclose(diag_grad, finite_difference, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_sample_uses_observation_time(dtype):
    model = ExtendedStateSpaceModel(
        transition_fn=lambda z, t, t_next, args: tfd.MultivariateNormalDiag(z, jnp.ones_like(z)),
        observation_fn=lambda z, t, args: tfd.MultivariateNormalDiag(jnp.full_like(z, t), jnp.full_like(z, 1e-9)),
        initial_state_prior=tfd.MultivariateNormalDiag(jnp.zeros(1, dtype=dtype), jnp.ones(1, dtype=dtype)),
        dt=0.25,
    )
    samples = jax.jit(lambda key: model.sample(key, 5, t0=10.))(jax.random.key(19))
    for array in samples:
        assert array.dtype == dtype
    np.testing.assert_allclose(samples.observation[:, 0], samples.t, rtol=0., atol=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_time_dependent_smoother_matches_joint_gaussian_conditioning(dense):
    dt, t0 = 0.5, 2.
    observations = np.array([0.5, -0.3, 0.8, 0.4])
    times = t0 + np.arange(len(observations) + 1) * dt

    def coefficient(t, t_next):
        return 0.5 + 0.1 * t + 0.2 * (t_next - t)

    model = ExtendedStateSpaceModel(
        transition_fn=lambda z, t, t_next, args: tfd.MultivariateNormalTriL(coefficient(t, t_next) * z, jnp.array([[0.4]])),
        observation_fn=lambda z, t, args: tfd.MultivariateNormalTriL((1. + 0.2 * t) * z, jnp.array([[0.3]])),
        initial_state_prior=tfd.MultivariateNormalTriL(jnp.array([0.2]), jnp.array([[0.7]])),
        materialise_jacobians=dense,
        dt=dt,
    )
    # Construct the entire joint latent Gaussian independently of the RTS recursion.
    means = np.zeros(len(times))
    covariance = np.zeros((len(times), len(times)))
    means[0], covariance[0, 0] = 0.2, 0.7 ** 2
    for i in range(1, len(times)):
        a = coefficient(times[i - 1], times[i])
        means[i] = a * means[i - 1]
        covariance[i, :i] = a * covariance[i - 1, :i]
        covariance[:i, i] = covariance[i, :i]
        covariance[i, i] = a ** 2 * covariance[i - 1, i - 1] + 0.4 ** 2
    projection = np.eye(len(times))[1:] * (1. + 0.2 * times[1:])[:, None]
    obs_cov = projection @ covariance @ projection.T + 0.3 ** 2 * np.eye(len(observations))
    gain = np.linalg.solve(obs_cov, projection @ covariance).T
    expected_mean = means + gain @ (observations - projection @ means)
    expected_cov = covariance - gain @ projection @ covariance

    filtered = model.forward_filter(jnp.asarray(observations[:, None]), t0=t0)
    smoothed, prior = jax.jit(lambda result: model.backward_smooth(result, include_prior=True))(filtered)
    # RTS explicitly regularizes predicted covariances with 1e-6 diagonal jitter.
    np.testing.assert_allclose(smoothed.smoothed_mean[:, 0], expected_mean[1:], rtol=0., atol=3e-6)
    np.testing.assert_allclose(smoothed.smoothed_cov[:, 0, 0], np.diag(expected_cov)[1:], rtol=0., atol=3e-6)
    np.testing.assert_allclose(prior.mean[0], expected_mean[0], rtol=0., atol=3e-6)
    np.testing.assert_allclose(prior.covariance[0, 0], expected_cov[0, 0], rtol=0., atol=3e-6)
    np.testing.assert_array_equal(smoothed.t, times[1:])


@pytest.mark.parametrize("dense", [False, True])
def test_masked_nonfinite_values_have_finite_correct_gradients(dense):
    model = linear_model(diagonal=False, dense=dense)
    observations = jnp.array([[0.2, -0.1, 0.3], [jnp.nan, jnp.inf, -jnp.inf], [0.1, 0.2, 0.6]])
    mask = jnp.array([False, True, False])
    finite_observations = jnp.where(mask[:, None], 0., observations)
    value_and_grad = jax.jit(jax.vmap(
        jax.value_and_grad(lambda scale, obs: model.log_prob(obs, transition_args=(scale,), mask=mask)),
        in_axes=(0, None),
    ))
    expected = value_and_grad(jnp.array([0.8, 1.2]), finite_observations)
    actual = value_and_grad(jnp.array([0.8, 1.2]), observations)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
    obs_grad = jax.jit(jax.grad(lambda obs: model.log_prob(obs, transition_args=(1.,), mask=mask)))(observations)
    assert np.isfinite(obs_grad).all()
    np.testing.assert_array_equal(obs_grad[1], 0.)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_state_dtype_and_explicit_prediction_time(dtype):
    model = linear_model(dtype=dtype)
    initial = model.create_initial_filter_state()
    assert initial.t.dtype == dtype
    assert initial.log_cumulative_marginal_likelihood.dtype == dtype
    predicted = model.incremental_predict(initial, transition_args=(1.,), t_next=2.5)
    predicted = model.incremental_predict(predicted, transition_args=(1.,))
    np.testing.assert_array_equal(predicted.t, 3.5)
    filtered = jax.jit(model.forward_filter)(jnp.zeros((3, 3), dtype=dtype), transition_args=(1.,))
    for array in filtered:
        assert array.dtype == dtype
    reference = linear_model(dtype=jnp.float64).forward_filter(
        jnp.zeros((3, 3), dtype=jnp.float64), transition_args=(1.,))
    for actual, expected in zip(filtered, reference):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_explicit_double_precision_time_is_preserved():
    model = linear_model(dtype=jnp.float32)
    t0 = jnp.array(2 ** 30 + 0.5, dtype=jnp.float64)
    initial = model.create_initial_filter_state(t0=t0)
    np.testing.assert_array_equal(initial.t, t0)
    predicted = jax.jit(lambda state: model.incremental_predict(state, transition_args=(1.,)))(initial)
    np.testing.assert_array_equal(predicted.t, t0 + model.dt)
