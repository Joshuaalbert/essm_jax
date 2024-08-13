"""Extended Gaussian State Space Model."""

import dataclasses
from typing import Callable, NamedTuple, Tuple, Union, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from tensorflow_probability.substrates.jax.math import hpsd_solve

from essm_jax.jvp_op import JVPLinearOp

tfpd = tfp.distributions
tfb = tfp.bijectors


class SampleResult(NamedTuple):
    t: jax.Array  # [num_timesteps] The time indices
    latent: jax.Array  # [num_timesteps, latent_size] The latent states
    observation: jax.Array  # [num_timesteps, observation_size] The observed states


class FilterResult(NamedTuple):
    """
    Represents the result of a forward filtering pass.
    """
    t: jax.Array  # [num_timesteps] The time indices
    log_cumulative_marginal_likelihood: jax.Array  # [num_timesteps] The log marginal likelihood of each timestep
    filtered_mean: jax.Array  # [num_timesteps, latent_size] The mean of p(z[t] | x[:t])
    filtered_cov: jax.Array  # [num_timesteps, latent_size, latent_size] The covariance of p(z[t] | x[:t])
    predicted_mean: jax.Array  # [num_timesteps, latent_size] The mean of p(z[t+1] | x[:t])
    predicted_cov: jax.Array  # [num_timesteps, latent_size, latent_size] The covariance of p(z[t+1] | x[:t])
    observation_mean: jax.Array  # [num_timesteps, observation_size] The mean of p(x[t] | x[:t-1])
    observation_cov: jax.Array  # [num_timesteps, observation_size, observation_size] The covariance of p(x[t] | x[:t-1])


class SmoothingResult(NamedTuple):
    """
    Represents the result of a backward smoothing pass.
    """
    t: jax.Array  # [num_timesteps] The time indices
    smoothed_mean: jax.Array  # [num_timesteps, latent_size] The mean of p(z[t] | x[:T])
    smoothed_cov: jax.Array  # [num_timesteps, latent_size, latent_size] The covariance of p(z[t] | x[:T])
    smoothed_obs_mean: jax.Array  # [num_timesteps, observation_size] The mean of p(x[t] | x[:T])
    smoothed_obs_cov: jax.Array  # [num_timesteps, observation_size, observation_size] The covariance of p(x[t] | x[:T])


class InitialPrior(NamedTuple):
    mean: jax.Array  # [latent_size] The mean of the initial state
    covariance: jax.Array  # [latent_size, latent_size] The covariance of the initial state


def _efficient_add_scalar_diag(A: jax.Array, c: Union[jax.Array, float]) -> jax.Array:
    """
    Efficiently add a scalar to the diagonal of a matrix.

    Efficiently computes A + c * I, where A is a matrix and c is a scalar.

    Args:
        A: [n, n] array
        c: scalar

    Returns:
        [n, n] array
    """
    # It's more efficiency to set the diagonal directly rather than  materialise the I
    n = A.shape[-1]
    c = jnp.asarray(c, A.dtype)
    return A.at[jnp.diag_indices(n)].add(c, indices_are_sorted=True, unique_indices=True)


@dataclasses.dataclass(eq=False)
class ExtendedStateSpaceModel:
    """Extended State Space Model.

    Implements the Kalman and RTS equations for non-linear state space models,
    using linear approximations to the transition and observation functions.

    Args:
        transition_fn: A function that computes the state transition distribution
            p(z[t] | z[t-1], t). Must return a MultivariateNormalLinearOperator.
            Call signature is `transition_fn(z[t-1], t)`, where z[t-1] is the previous state.
        observation_fn: A function that computes the observation distribution
            p(x[t] | z[t], t). Must return a MultivariateNormalLinearOperator.
            Call signature is `observation_fn(z[t], t)`, where z[t] is the current state.
            Note: t in [1, num_time] is the observation time index, with t=0 being the initial state.
        initial_state_prior: A distribution over the initial state p(z[0]).
            Must be a MultivariateNormalLinearOperator.
        more_data_than_params: If True, the observation function has more outputs than inputs.
        materialise_jacobians: If True, the Jacobians are materialised as dense matrices.
    """
    transition_fn: Callable[[jax.Array, jax.Array], tfpd.MultivariateNormalLinearOperator]
    observation_fn: Callable[[jax.Array, jax.Array], tfpd.MultivariateNormalLinearOperator]
    initial_state_prior: tfpd.MultivariateNormalLinearOperator
    more_data_than_params: bool = False
    materialise_jacobians: bool = False

    def __post_init__(self):
        if not callable(self.transition_fn):
            raise ValueError('`transition_fn` must be a callable.')
        if not callable(self.observation_fn):
            raise ValueError('`observation_fn` must be a callable.')
        if not isinstance(self.initial_state_prior, tfpd.MultivariateNormalLinearOperator):
            raise ValueError('`initial_state_prior` must be a `MultivariateNormalLinearOperator` '
                             'instance.')

        _initial_state_prior_mean = self.initial_state_prior.mean()
        self.latent_size = np.size(_initial_state_prior_mean)
        self.latent_shape = np.shape(_initial_state_prior_mean)

    def get_transition_jacobian(self, t: jax.Array) -> JVPLinearOp:
        def _transition_fn(z):
            return self.transition_fn(z, t).mean()

        return JVPLinearOp(_transition_fn, more_outputs_than_inputs=False)

    def get_observation_jacobian(self, t: jax.Array, observation_size: Optional[int] = None) -> JVPLinearOp:
        def _observation_fn(z):
            return self.observation_fn(z, t).mean()

        more_data_than_params = self.more_data_than_params
        if observation_size is not None:
            more_data_than_params = self.latent_size < observation_size
        return JVPLinearOp(_observation_fn, more_outputs_than_inputs=more_data_than_params)

    def transition_matrix(self, z, t):
        Fop = self.get_transition_jacobian(t)
        return Fop(z).to_dense()

    def observation_matrix(self, z, t):
        Hop = self.get_observation_jacobian(t)
        return Hop(z).to_dense()

    def sample(self, key, num_time: int, t0: Union[jax.Array, int] = 0) -> SampleResult:
        """
        Sample from the model.

        Args:
            key: a PRNGKey
            num_time: the number of time steps to sample
            t0: the time of initial state

        Returns:
            latents: [num_time, latent_size] array of latents
            observables: [num_time, observation_size] array of observables
        """
        init_key, latent_key = jax.random.split(key, 2)

        # Binary operation: S -> (S, O)
        # transition: S -> S
        # observation: S -> O

        def _sample_latents_op(latent, y):
            (key, t) = y
            new_latent_key, obs_key = jax.random.split(key, 2)
            transition_dist = self.transition_fn(latent, t)
            new_latent = transition_dist.sample(seed=new_latent_key)
            observation_dist = self.observation_fn(new_latent, t)
            new_observation = observation_dist.sample(seed=obs_key)
            return new_latent, SampleResult(t=t, latent=new_latent, observation=new_observation)

        # Sample at t0
        init = self.initial_state_prior.sample(seed=init_key)
        xs = (
            jax.random.split(latent_key, num_time),
            jnp.arange(1, num_time + 1) + t0
        )
        _, samples = lax.scan(
            _sample_latents_op,
            init=init,
            xs=xs
        )

        return samples

    def _check_shapes(self, observations: jax.Array, mask: Optional[jax.Array] = None):
        """
        Check the shapes of the observations and mask.

        Args:
            observations: [num_time, observation_size] array of observations
            mask: [num_time] array of masks, True for missing observations

        Raises:
            ValueError: If the shapes are incorrect.
        """

        if len(observations.shape) != 2:
            raise ValueError('observations must be a 2D array, of shape [num_time, observation_size]')

        if mask is not None:
            if len(mask.shape) != 1:
                raise ValueError('mask must be a 1D array, of shape [num_time]')
            if mask.shape[0] != observations.shape[0]:
                raise ValueError('mask and observations must have the same length')

    def forward_simulate(self, key: jax.Array, num_time: int,
                         observations: jax.Array, mask: Optional[jax.Array] = None) -> SampleResult:
        """
        Simulate from the model, from the end of the forward filtering pass.

        Args:
            key: a PRNGKey
            num_time: the number of time steps to simulate
            observations: [num_time, observation_size] array of observations
            mask: [num_time] array of masks, True for missing observations

        Returns:
            sample result: num_time timesteps of forward simulated
        """
        filter_result = self.forward_filter(observations, mask)
        initial_state_prior = tfpd.MultivariateNormalTriL(
            loc=filter_result.filtered_mean[-1],
            scale_tril=jnp.linalg.cholesky(_efficient_add_scalar_diag(filter_result.filtered_cov[-1], 1e-6))
        )
        new_essm = ExtendedStateSpaceModel(
            transition_fn=self.transition_fn,
            observation_fn=self.observation_fn,
            initial_state_prior=initial_state_prior,
            more_data_than_params=self.more_data_than_params,
            materialise_jacobians=self.materialise_jacobians
        )
        return new_essm.sample(key=key, num_time=num_time, t0=filter_result.t[-1])

    def forward_filter(self, observations: jax.Array, mask: Optional[jax.Array] = None,
                       marginal_likelihood_only: bool = False,
                       t0: Union[jax.Array, int] = 0) -> Union[FilterResult, jax.Array]:
        """
        Run the forward filtering pass, computing the total marginal likelihood

        p(x) = prod_t p(x[t] | x[:t-1])

        filtered latent distribution at each timestep,

        p(z[t] | x[:t])

        Args:
            observations: [num_time, observation_size] array of observations
            mask: [num_time] array of masks, True for missing observations
            marginal_likelihood_only: if True only the marginal likelihood is returned
            t0: the time of initial state

        Returns:
            If `marginal_likelihood_only` is True, the log marginal likelihood of the observations.
            Otherwise, a `FilterResult` instance, with [num_time] batched arrays.
        """
        self._check_shapes(observations=observations, mask=mask)

        class Carry(NamedTuple):
            log_cumulative_marginal_likelihood: jax.Array  # log marginal likelihood
            predicted_mean: jax.Array  # [latent_size] mean of p(z[t+1] | x[:t])
            predicted_cov: jax.Array  # [latent_size, latent_size] covariance of p(z[t+1] | x[:t])

        class YType(NamedTuple):
            observation: jax.Array  # [observation_size] observation at time t
            mask: jax.Array  # mask at time t
            t: jax.Array  # time index

        num_time = np.shape(observations)[0]

        def _filter_op(carry: Carry, y: YType) -> Tuple[Carry, FilterResult]:
            """
            A single step of the forward equations.
            """

            # Note: We perform update FIRST, then predict which is contrary to the usual order.
            # This is so that the filter results naturally align with the smoothing operation.

            Hop = self.get_observation_jacobian(t=y.t, observation_size=y.observation.size)
            H = Hop(carry.predicted_mean)
            if self.materialise_jacobians:
                H = H.to_dense()

            # Update step, compute p(z[t] | x[:t]) from p(z[t] | x[:t-1])
            observation_dist = self.observation_fn(carry.predicted_mean, y.t)
            R = observation_dist.covariance()
            # Push-forward the prior (i.e. predictive) distribution to the observation space
            x_expectation = observation_dist.mean()  # [observation_size]
            tmp_H_P = H @ carry.predicted_cov
            S = tmp_H_P @ H.T + R  # [observation_size, observation_size]
            S_chol = jnp.linalg.cholesky(S)  # [observation_size, observation_size]

            marginal_dist = tfpd.MultivariateNormalTriL(x_expectation, S_chol)

            # Compute the log marginal likelihood p(x[t] | x[:t-1])
            log_marginal_likelihood = marginal_dist.log_prob(y.observation)

            # Compute the Kalman gain
            # K = predict_cov @ H.T @ inv(S) = predict_cov.T @ H.T @ inv(S)
            K = hpsd_solve(S, tmp_H_P, cholesky_matrix=S_chol).T  # [latent_size, observation_size]

            # Update the state estimate
            filtered_mean = carry.predicted_mean + K @ (y.observation - x_expectation)  # [latent_size]

            # Update the state covariance using Joseph's form to ensure positive semi-definite
            # tmp_factor = (I - K @ H)
            if self.more_data_than_params:
                tmp_factor = _efficient_add_scalar_diag((- K) @ H, 1.)
            else:
                tmp_factor = _efficient_add_scalar_diag(K @ (-H), 1.)
            filtered_cov = tmp_factor @ carry.predicted_cov @ tmp_factor.T + K @ R @ K.T  # [latent_size, latent_size]

            # When masked, then the filtered state is the predicted state.
            if mask is not None:
                filtered_mean = lax.select(*jnp.broadcast_arrays(y.mask,
                                                                 carry.predicted_mean, filtered_mean))
                filtered_cov = lax.select(*jnp.broadcast_arrays(y.mask,
                                                                carry.predicted_cov, filtered_cov))
                log_marginal_likelihood = lax.select(*jnp.broadcast_arrays(y.mask,
                                                                           jnp.zeros_like(log_marginal_likelihood),
                                                                           log_marginal_likelihood))

            # Predict step, compute p(z[t+1] | x[:t])

            Fop = self.get_transition_jacobian(t=y.t + 1)
            F = Fop(filtered_mean)
            if self.materialise_jacobians:
                F = F.to_dense()

            predicted_dist = self.transition_fn(filtered_mean, y.t + 1)
            predicted_mean = predicted_dist.mean()  # [latent_size]
            Q = predicted_dist.covariance()  # [latent_size, latent_size]
            predicted_cov = F @ filtered_cov @ F.T + Q  # [latent_size, latent_size]

            # Update cumulative marginal likelihood
            log_cumulative_marginal_likelihood = carry.log_cumulative_marginal_likelihood + log_marginal_likelihood

            return Carry(
                log_cumulative_marginal_likelihood=log_cumulative_marginal_likelihood,
                predicted_mean=predicted_mean,
                predicted_cov=predicted_cov
            ), FilterResult(
                t=y.t,
                log_cumulative_marginal_likelihood=log_cumulative_marginal_likelihood,
                filtered_mean=filtered_mean,
                filtered_cov=filtered_cov,
                predicted_mean=predicted_mean,
                predicted_cov=predicted_cov,
                observation_mean=x_expectation,
                observation_cov=S
            )

        # Push forward initial state to create p(z[1] | z[0])
        t0 = jnp.asarray(t0, jnp.int32)
        t1 = t0 + jnp.asarray(1, jnp.int32)
        init_predict_dist = self.transition_fn(self.initial_state_prior.mean(), t1)

        init_Fop = self.get_transition_jacobian(t1)
        init_F = init_Fop(self.initial_state_prior.mean())
        if self.materialise_jacobians:
            init_F = init_F.to_dense()
        init_predicted_mean = init_predict_dist.mean()
        init_predicted_cov = init_F @ self.initial_state_prior.covariance() @ init_F.T + init_predict_dist.covariance()

        # init_predicted_mean = self.initial_state_prior.mean()
        # init_predicted_cov = self.initial_state_prior.covariance()
        init_result = Carry(
            log_cumulative_marginal_likelihood=jnp.asarray(0.),
            predicted_mean=init_predicted_mean,
            predicted_cov=init_predicted_cov
        )
        if mask is None:
            _mask = jnp.zeros(num_time, dtype=jnp.bool_)  # dummy variable (we skip the mask select)
        else:
            _mask = mask
        xs = YType(
            observation=observations,
            mask=_mask,
            t=jnp.arange(1, num_time + 1, dtype=jnp.int32) + t0
        )
        final_accumulate, filter_results = lax.scan(
            f=_filter_op,
            init=init_result,
            xs=xs
        )
        if marginal_likelihood_only:
            return final_accumulate.log_cumulative_marginal_likelihood
        return filter_results

    def log_prob(self, observations: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        """
        Compute the log probability of the observations under the model.

        Args:
            observations: [num_time, observation_size] array of observations
            mask: [num_time] array of masks, True for missing observations

        Returns:
            [num_time] array of log probabilities
        """
        return self.forward_filter(observations, mask, marginal_likelihood_only=True)

    def posterior_marginals(self, observations: jax.Array, mask: Optional[jax.Array] = None,
                            t0: Union[jax.Array, int] = 0) -> Union[
        SmoothingResult, Tuple[SmoothingResult, InitialPrior]]:
        """
        Compute the posterior marginal distributions of the latents, p(z[t] | x[:T]).

        Args:
            observations: [num_time, observation_size] array of observations
            mask: [num_time] array of masks, True for missing observations
            t0: the time of initial state

        Returns:
            A `SmoothingResult` instance, with [num_time] batched arrays.
        """
        filter_result = self.forward_filter(observations, mask, t0=t0)
        return self.backward_smooth(filter_result, include_prior=False)

    def backward_smooth(self, filter_result: FilterResult, include_prior: bool = False) -> Union[
        SmoothingResult, Tuple[SmoothingResult, InitialPrior]]:
        """
        Run the backward smoothing pass.

        Args:
            filter_result: A `FilterResult` instance, with [num_time] batched arrays.
            include_prior: if True, include the prior p(z0) in the smoothing pass.

        Returns:
            A `SmoothingResult` instance, with [num_time] batched arrays.
            and, if include_prior is True then also an `InitialPrior` instance.
        """

        class Carry(NamedTuple):
            smoothed_mean: jax.Array  # [latent_size] mean of p(z[t] | x[:T])
            smoothed_cov: jax.Array  # [latent_size, latent_size] covariance of p(z[t] | x[:T])

        init_carry = Carry(
            smoothed_mean=filter_result.predicted_mean[-1],  # [latent_size] mean of p(z[T] | x[:T])
            smoothed_cov=filter_result.predicted_cov[-1]  # [latent_size, latent_size] covariance of p(z[T] | x[:T])
        )

        def _smooth_op(carry: Carry, y: FilterResult) -> Tuple[Carry, SmoothingResult]:
            """
            A single step of the backward equations.
            """
            Fop = self.get_transition_jacobian(t=y.t)
            F = Fop(y.filtered_mean)
            if self.materialise_jacobians:
                F = F.to_dense()

            # Compute the RTS smoother gain
            # J = y.filtered_cov @ F.T @ jnp.linalg.inv(y.predicted_cov)
            predicted_cov_chol = jnp.linalg.cholesky(y.predicted_cov)  # Possibly need to add a small diagonal jitter
            tmp_F_P = F @ y.filtered_cov
            J = hpsd_solve(y.predicted_cov, tmp_F_P, cholesky_matrix=predicted_cov_chol).T

            # Update the state estimate
            smoothed_mean = y.filtered_mean + J @ (carry.smoothed_mean - y.predicted_mean)

            # Update the state covariance
            smoothed_cov = y.filtered_cov + J @ (carry.smoothed_cov - y.predicted_cov) @ J.T

            # Push-forward the smoothed distribution to the observation space
            observation_dist = self.observation_fn(smoothed_mean, y.t)
            R = observation_dist.covariance()

            Hop = self.get_observation_jacobian(t=y.t, observation_size=y.observation_mean.size)
            H = Hop(smoothed_mean)
            if self.materialise_jacobians:
                H = H.to_dense()
            smoothed_obs_mean = observation_dist.mean()
            smoothed_obs_cov = H @ smoothed_cov @ H.T + R

            return Carry(
                smoothed_mean=smoothed_mean,
                smoothed_cov=smoothed_cov
            ), SmoothingResult(
                t=y.t,
                smoothed_mean=smoothed_mean,
                smoothed_cov=smoothed_cov,
                smoothed_obs_mean=smoothed_obs_mean,
                smoothed_obs_cov=smoothed_obs_cov
            )

        final_carry, smooth_results = lax.scan(
            f=_smooth_op,
            init=init_carry,
            xs=filter_result,
            reverse=True
        )

        if include_prior:
            # Transition prior to compute predicted p(z1 | z0)
            t1 = filter_result.t[0]
            init_predict_dist = self.transition_fn(self.initial_state_prior.mean(), t1)
            init_Fop = self.get_transition_jacobian(t=t1)
            init_F = init_Fop(self.initial_state_prior.mean())
            if self.materialise_jacobians:
                init_F = init_F.to_dense()
            init_predicted_mean = init_predict_dist.mean()
            init_predicted_cov = init_F @ self.initial_state_prior.covariance() @ init_F.T + init_predict_dist.covariance()

            t0 = t1 - jnp.asarray(1, jnp.int32)
            y = FilterResult(
                t=t0,
                log_cumulative_marginal_likelihood=jnp.asarray(0.),
                filtered_mean=self.initial_state_prior.mean(),
                filtered_cov=self.initial_state_prior.covariance(),
                predicted_mean=init_predicted_mean,
                predicted_cov=init_predicted_cov,
                observation_mean=filter_result.observation_mean[0],  # dummy unused
                observation_cov=filter_result.observation_cov[0]  # dummy unused
            )
            final_initial_prior_carry, _ = _smooth_op(
                carry=final_carry,
                y=y
            )
            smoothed_prior = InitialPrior(
                mean=final_initial_prior_carry.smoothed_mean,
                covariance=final_initial_prior_carry.smoothed_cov
            )
            return smooth_results, smoothed_prior

        return smooth_results
