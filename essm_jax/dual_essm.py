import dataclasses
from typing import NamedTuple, Optional, Tuple

import jax
import tensorflow_probability.substrates.jax as tfp

from essm_jax.essm import ExtendedStateSpaceModel, IncrementalFilterState

tfpd = tfp.distributions

__all__ = [
    'DualExtendedStateSpaceModel',
    'DualIncrementalFilterState'
]


class DualIncrementalFilterState(NamedTuple):
    state_filter_state: IncrementalFilterState
    param_filter_state: IncrementalFilterState


@dataclasses.dataclass(eq=False)
class DualExtendedStateSpaceModel:
    """
    Dual extended state space model for simulatanous state and parameter estimation, in an online setup. For iterative
    estimation, the normal ESSM can be used effectively.

    Args:
        state_essm: the state ESSM, should accept the parameter as an input
        param_essm: the parameter ESSM, should accept the state as an input
    """
    state_essm: ExtendedStateSpaceModel
    param_essm: ExtendedStateSpaceModel

    def create_initial_filter_state(self, t0: jax.Array | float = 0.0) -> DualIncrementalFilterState:
        return DualIncrementalFilterState(
            state_filter_state=self.state_essm.create_initial_filter_state(t0),
            param_filter_state=self.param_essm.create_initial_filter_state(t0)
        )

    def incremental_update(self, filter_state: DualIncrementalFilterState, observation: jax.Array,
                           mask: Optional[jax.Array] = None) -> Tuple[
        DualIncrementalFilterState, tfpd.MultivariateNormalLinearOperator]:
        """
        Incremental update of the dual ESSM.

        Args:
            filter_state: the current filter state
            observation: the observation at time t
            mask: the mask for the observation

        Returns:
            the updated filter state and the marginal distribution
        """
        state_filter_state, marginal_dist = self.state_essm.incremental_update(
            filter_state.state_filter_state,
            observation,
            filter_state.param_filter_state.filtered_mean,
            mask=mask
        )
        param_filter_state, _ = self.param_essm.incremental_update(
            filter_state.param_filter_state,
            observation,
            state_filter_state.filtered_mean,
            mask=mask
        )
        return (
            DualIncrementalFilterState(state_filter_state=state_filter_state, param_filter_state=param_filter_state),
            marginal_dist
        )

    def incremental_predict(self, filter_state: DualIncrementalFilterState) -> DualIncrementalFilterState:
        """
        Incremental prediction of the dual ESSM.

        Args:
            filter_state: the current filter state

        Returns:
            the updated filter state
        """
        state_theta = filter_state.param_filter_state.filtered_mean
        param_theta = filter_state.state_filter_state.filtered_mean
        state_filter_state = self.state_essm.incremental_predict(filter_state.state_filter_state, state_theta)
        param_filter_state = self.param_essm.incremental_predict(filter_state.param_filter_state, param_theta)
        return DualIncrementalFilterState(state_filter_state=state_filter_state, param_filter_state=param_filter_state)
