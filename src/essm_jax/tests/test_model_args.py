"""Separate callback arguments must survive every inference and sampling path."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from essm_jax.essm import ExtendedStateSpaceModel, convert_filter_state_to_incremental_state

tfd = tfp.distributions


class ModelCase(NamedTuple):
    model: ExtendedStateSpaceModel
    reference: ExtendedStateSpaceModel
    transition_args: tuple
    observation_args: tuple
    observations: jax.Array


def _model_case(dense: bool) -> ModelCase:
    parameters = {
        "matrix": jnp.array([[0.7, 0.2], [0., 0.85]]),
        "drift": jnp.array([0.1, -0.2]),
        "scale": jnp.array([0.3, 0.4]),
    }
    projection = jnp.array([[1., 0.2], [-0.3, 0.7], [0.5, 1.]])
    bias = jnp.array([0.1, -0.2, 0.3])
    noise = jnp.array([0.4, 0.7, 0.5])

    def transition(z, t, t_next, args):
        params, = args
        return tfd.MultivariateNormalDiag(
            params["matrix"] @ z + params["drift"] * (t_next - t), params["scale"]
        )

    def observation(z, t, args):
        matrix, offset, scale = args
        return tfd.MultivariateNormalDiag(matrix @ z + offset * t, scale)

    # Bind the same model independently of the new argument-routing path.
    # These callbacks also check that the default empty tuple is passed intact.
    def bound_transition(z, t, t_next, args):
        assert args == ()
        return tfd.MultivariateNormalDiag(
            parameters["matrix"] @ z + parameters["drift"] * (t_next - t), parameters["scale"]
        )

    def bound_observation(z, t, args):
        assert args == ()
        return tfd.MultivariateNormalDiag(projection @ z + bias * t, noise)

    prior = tfd.MultivariateNormalDiag(jnp.array([0.2, -0.1]), jnp.array([0.8, 1.1]))
    return ModelCase(
        model=ExtendedStateSpaceModel(transition, observation, prior, materialise_jacobians=dense, dt=0.4),
        reference=ExtendedStateSpaceModel(bound_transition, bound_observation, prior,
                                          materialise_jacobians=dense, dt=0.4),
        transition_args=(parameters,),
        observation_args=(projection, bias, noise),
        observations=jnp.array([[0.2, -0.1, 0.3], [0.7, 0.4, -0.5], [0.1, 0.2, 0.6]]),
    )


@pytest.mark.parametrize("dense", [False, True])
def test_separate_args_route_through_filter_smoother_and_jacobians(dense):
    case = _model_case(dense)
    model, reference = case.model, case.reference
    transition_args, observation_args = case.transition_args, case.observation_args
    mask, t0 = jnp.array([False, True, False]), 1.5
    filtered = jax.jit(model.forward_filter)(
        case.observations, transition_args=transition_args, observation_args=observation_args, mask=mask, t0=t0
    )
    expected = reference.forward_filter(case.observations, mask=mask, t0=t0)
    comparisons = [(filtered, expected)]

    state = model.create_initial_filter_state(t0)
    for i, observation in enumerate(case.observations):
        state = model.incremental_predict(state, transition_args=transition_args)
        state, marginal = model.incremental_update(state, observation, observation_args=observation_args, mask=mask[i])
        comparisons.extend([
            (state.filtered_mean, expected.filtered_mean[i]),
            (state.filtered_cov, expected.filtered_cov[i]),
            (state.t, expected.t[i]),
            (state.log_cumulative_marginal_likelihood, expected.log_cumulative_marginal_likelihood[i]),
            (marginal.mean(), expected.observation_mean[i]),
            (marginal.covariance(), expected.observation_cov[i]),
        ])

    smoothed = jax.jit(lambda result, tr, obs: model.backward_smooth(
        result, transition_args=tr, observation_args=obs, include_prior=True
    ))(filtered, transition_args, observation_args)
    expected_smoothed = reference.backward_smooth(expected, include_prior=True)
    comparisons.append((smoothed, expected_smoothed))
    comparisons.append((model.posterior_marginals(
        case.observations, transition_args=transition_args, observation_args=observation_args, mask=mask, t0=t0
    ), expected_smoothed[0]))
    comparisons.append((model.log_prob(
        case.observations, transition_args=transition_args, observation_args=observation_args, mask=mask, t0=t0
    ), expected.log_cumulative_marginal_likelihood[-1]))

    mean = state.filtered_mean
    comparisons.extend([
        (model.transition_matrix(mean, state.t, state.t + model.dt, transition_args=transition_args),
         transition_args[0]["matrix"]),
        (model.get_transition_jacobian(state.t, state.t + model.dt,
                                       transition_args=transition_args)(mean).to_dense(), transition_args[0]["matrix"]),
        (model.observation_matrix(mean, state.t, observation_args=observation_args), observation_args[0]),
        (model.get_observation_jacobian(state.t, observation_args=observation_args,
                                        observation_size=3)(mean).to_dense(), observation_args[0]),
    ])
    for actual, expected in comparisons:
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-11, atol=1e-11)


def test_separate_args_route_through_sampling_and_both_forecast_inputs():
    case = _model_case(False)
    key = jax.random.key(31)
    samples = jax.jit(lambda key, tr, obs: case.model.sample(
        key, 4, transition_args=tr, observation_args=obs, t0=2.
    ))(key, case.transition_args, case.observation_args)
    comparisons = [(samples, case.reference.sample(key, 4, t0=2.))]

    filtered = case.model.forward_filter(case.observations, transition_args=case.transition_args,
                                         observation_args=case.observation_args, t0=2.)
    expected = case.reference.forward_filter(case.observations, t0=2.)
    terminal = convert_filter_state_to_incremental_state(filtered)
    for filter_input in (filtered, terminal):
        forecast = jax.jit(lambda key, state, tr, obs: case.model.forward_simulate(
            key, 4, state, transition_args=tr, observation_args=obs
        ))(key, filter_input, case.transition_args, case.observation_args)
        comparisons.append((forecast, case.reference.forward_simulate(key, 4, expected)))

    for actual, expected in comparisons:
        for actual_leaf, expected_leaf in zip(actual, expected):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("dense", [False, True])
def test_nested_args_support_jitted_batched_gradients_for_both_callbacks(dense):
    case = _model_case(dense)
    params, = case.transition_args
    projection, bias, noise = case.observation_args

    def likelihood(transition_scale, observation_scale):
        return case.model.log_prob(
            case.observations,
            transition_args=({**params, "scale": params["scale"] * transition_scale},),
            observation_args=(projection, bias, noise * observation_scale),
        )

    scales = jnp.array([[0.7, 1.2], [1.1, 0.85]])
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(likelihood, argnums=(0, 1))))(
        scales[:, 0], scales[:, 1]
    )
    assert np.isfinite(values).all()
    step = 1e-5
    for i, (transition_scale, observation_scale) in enumerate(scales):
        expected_transition_grad = (
            likelihood(transition_scale + step, observation_scale)
            - likelihood(transition_scale - step, observation_scale)
        ) / (2 * step)
        expected_observation_grad = (
            likelihood(transition_scale, observation_scale + step)
            - likelihood(transition_scale, observation_scale - step)
        ) / (2 * step)
        assert abs(expected_transition_grad) > 1e-3
        assert abs(expected_observation_grad) > 1e-3
        np.testing.assert_allclose(gradients[0][i], expected_transition_grad, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(gradients[1][i], expected_observation_grad, rtol=1e-7, atol=1e-7)
