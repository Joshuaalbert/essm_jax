# ESSM invariants

These are contracts to preserve when changing the state-space model, its JAX
operators, or its development tooling. Update this file and the corresponding
tests together when a contract deliberately changes. Numerical comparisons use
dtype-appropriate tolerances; they are not promises of bitwise equivalence.

## State, shapes, and time

- Observations passed to `forward_filter` have shape `[num_time, observation_size]`.
  A supplied mask has shape `[num_time]` and the same length. Invalid ranks or
  lengths raise `ValueError`.
- Result containers retain their NamedTuple structure. Means have a trailing
  latent or observation dimension, and covariances have two such dimensions.
- `create_initial_filter_state(t0)` represents the prior at `t0`, with zero
  accumulated log likelihood. `incremental_predict` advances the time to
  `t_next`, or by `dt` when no explicit next time is supplied.
- Python scalar times follow the prior's floating-point dtype. Explicit
  higher-precision timestamps retain their precision. The initial log likelihood
  uses the prior's dtype, and explicit prediction times remain JAX scalars.
- `incremental_update` incorporates the observation at the current time without
  advancing time. `forward_filter` predicts once from the prior, then updates
  and predicts at each observation. Its observation times are
  `t0 + (arange(num_time) + 1) * dt`.
- `FilterResult.predicted_mean` and `predicted_cov` describe the next state after
  the update on the same row. Conversion to an incremental state uses the last
  *filtered* state and its accumulated log likelihood.

## Inference and missing data

- Batch filtering agrees with applying the incremental predict/update API in
  the documented order, including masks and parameters passed through `*theta`.
- `mask=True` means the observation is missing: the update preserves the incoming
  state mean/covariance and adds zero log likelihood. The transition still runs.
  Masked NaN/Inf observations do not contaminate values or derivatives; their
  observation gradients are zero, including under `jit` and `vmap`.
- `log_prob` and `forward_filter(..., marginal_likelihood_only=True)` return the
  same scalar total as the last cumulative log likelihood in a nonempty full
  filter result.
- On well-conditioned linear Gaussian models, filtering agrees with the Kalman
  equations. The dense and JVP implementations produce equivalent filtering
  and smoothing results within numerical tolerance.
- Diagonal Gaussian scales are standard deviations: their squared values enter
  covariance propagation, Joseph updates, and observation marginals. Equivalent
  diagonal and triangular scale representations agree in values and gradients.
- Smoothing preserves observation times and ends at the final filtered latent
  distribution. `include_prior=True` returns the smoothed prior separately,
  without adding a row to the returned observation-time results.
  A row's smoothing Jacobian is evaluated over its prediction interval, from
  `t` to `t + dt`, including for time-dependent transition callbacks.
- Covariance calculations retain their matrix shapes and produce finite,
  symmetric, positive-semidefinite results for the well-conditioned regression
  models. Preserve the Joseph-form update and explicit numerical jitter.
- Forward simulation begins after the final filtered time. Equivalent batch
  and incremental terminal states give the same samples for the same PRNG key.

## JAX and operators

- Random sampling takes an explicit PRNG key. Reusing the same inputs and key
  produces reproducible results.
  Sampled observations use the newly sampled state's time, matching the returned
  timestamp and the filtering time convention.
- The tested filtering/operator paths remain usable under `jax.jit` and
  `jax.vmap`; do not convert traced array values into Python control flow.
- `materialise_jacobians=False` uses JVP/VJP products. Do not replace these with
  dense Jacobian construction in the hot path without performance evidence.
- For real array inputs, `JVPLinearOp` products agree with the dense Jacobian;
  transpose products agree with its transpose. The linearized and direct paths
  agree, including derivatives through the products. `.T` applies to `matvec`,
  `matmul`, and `to_dense` as well as `@`; transposing twice restores the original
  operator. Dense Jacobians include every positional input argument. Structured
  tangents retain their pytree structure.
- `pytree_unravel` restores tree structure, leaf shapes, and original leaf dtypes
  after flattening compatible inputs into a common dtype.
  Incompatible structures/shapes and incorrect flat lengths raise `ValueError`
  during tracing. Empty trees round-trip through an empty float32 vector.
- The dual filter updates the state using the old parameter estimate, then
  updates parameters using the new state estimate. Prediction uses both input
  estimates; it does not use one newly predicted estimate to predict the other.

## Development and release

- Local Python commands run in the `essm_py` conda environment.
- Installing `.[tests]` installs the package and its test/lint/build tools.
  Both wheels and source distributions retain the runtime and extras metadata.
- CI runs deterministic tests on CPU with double precision enabled. Timing
  comparisons remain available through the `benchmark` marker and do not gate
  correctness on shared runners.
- Coverage measures lines and branches across `src/essm_jax`, including the dual
  model and unexecuted production files. Only tests are omitted. The minimum
  combined coverage is 80%; reports include missing lines and XML/HTML output.
- Distribution builds and metadata checks must pass before manual publication.
  Build and test jobs do not publish packages or require publishing credentials.

The regression suite is under `src/essm_jax/tests/`. These contracts describe the
tested API; they do not establish correctness for every nonlinear approximation,
singular covariance, or complex-valued model.
