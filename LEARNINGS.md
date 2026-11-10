# Repository learnings

Keep this file focused on verified, reusable facts about ESSM. Update entries
when the implementation changes; do not copy another project's numerical notes.

## Development setup

- The local environment is `essm_py`. Conda supplies Python; pip inside the
  environment supplies project dependencies.
- `pyproject.toml` is the sole packaging and dependency definition. Runtime
  requirements and the `tests`/`examples` extras are static metadata; neither
  `setup.py` nor requirements files are needed to build a wheel from an sdist.
- The earlier `numpy<2` constraint prevented installation on Python 3.14. The
  dependency setup now follows JAXNS's use of NumPy and `tfp-nightly` with JAX.
  The import namespace is still `tensorflow_probability.substrates.jax`;
  TensorFlow itself is not needed for the JAX substrate.
- The default test run excludes the hardware-dependent speed comparison. Run
  it explicitly with `conda run -n essm_py pytest -m benchmark`.
- Coverage is opt-in locally (`pytest --cov`) and required in CI. Its source
  selection and 80% minimum live in `pyproject.toml`.

## Numerical conventions

- ESSM places the first observation one transition after the initial prior.
  A comparison with another Kalman implementation must align that first prior
  and time index before comparing results.
- Filter result rows contain an updated state and the prediction for the next
  step. Repeated incremental calls must use the same ordering as batch filtering.
- RTS smoothing must differentiate the transition from a row's time to its next
  time. Using the preceding interval gives incorrect results for time-dependent
  transitions even when time-independent tests pass.
- TFP `MultivariateNormalDiag.scale.diag_part()` contains standard deviations.
  Square these entries when adding process or observation noise to a covariance.
- Selecting the final update with a mask does not protect reverse-mode gradients
  from NaN/Inf residuals. Replace missing observations before residual and
  likelihood arithmetic, then select the unchanged state and zero likelihood.
- `jax.linearize` already returns a linear tangent map. Reuse that map for JVPs
  and its `linear_transpose` for VJPs; a transpose view can share the residuals.
- A likelihood-only scan should return `None` for its per-step outputs. Otherwise
  an unjitted caller allocates the filter history before discarding it; an outer
  `jit` can hide this allocation by eliminating the unused outputs.
- The public helper module is named `pytee_utils.py` (historical spelling).
  Preserve that import path when doing unrelated work.

## Performance measurements (2026-11-10)

`benchmarks/filtering.py` compares filtering, scalar likelihood, and its process
noise-scale derivative using the same nonlinear, time-independent model. The
baseline is commit `a4a8b7b`. Both versions use triangular noise scales so the
diagonal-noise bug does not invalidate the performance reference. Values and
gradients agree at `rtol=atol=1e-10` on the measured float64 CPU cases.

Environment: aarch64 Linux, Python 3.14.10, JAX/jaxlib 0.12.0.1, CPU device 0,
x64 enabled. Each phase uses five cache-cleared lowering/compilation repetitions
and nine synchronized execution repetitions after warming the compiled call.
Reported intervals are the 25th–75th percentiles; these are local CPU results.

For 256 steps with 64 latent and 64 observation variables (milliseconds):

| Operation | Lowering before → after, median [IQR] | Compilation before → after, median [IQR] | Execution before → after, median [IQR] |
| --- | --- | --- | --- |
| Filter | 83.4 [82.9–85.5] → 66.2 [65.6–67.7] | 104.2 [103.9–104.6] → 102.5 [101.9–103.2] | 50.7 [49.5–52.4] → 50.5 [50.3–50.9] |
| Log probability | 78.9 [77.3–79.8] → 59.8 [59.7–60.6] | 77.8 [77.7–78.9] → 75.6 [75.3–78.8] | 42.2 [41.5–53.0] → 41.9 [41.9–46.0] |
| Value and gradient | 198.5 [198.2–199.5] → 165.1 [163.9–165.6] | 197.9 [196.9–200.6] → 194.2 [193.6–194.9] | 139.6 [138.1–141.4] → 136.0 [135.1–136.6] |

The main timing improvement is tracing/lowering (17–24%). Compiled execution is
similar. HLO inspection shows the matrix operations remain; the compiler already
eliminated much of the duplicated linearization work. No device speedup is assumed
from the Python simplification alone.

For the same shape, the likelihood scan's output storage, calculated from its
JAXPR shapes/dtypes, falls from 25,595,404 bytes to 33,296 bytes (24.4 MiB to
32.5 KiB). This is not a measured process-memory peak: it describes the history
avoided when calling without an outer `jit`. With an outer `jit`, dead-output
elimination already removes the history; compiler-reported likelihood temporary
storage stays approximately 302 KB. Gradient temporary storage is still about
145 MB, and reverse-mode differentiation still saves intermediates over time.

For 1,000 steps with 128 latent and 128 observation variables (milliseconds):

| Operation | Lowering before → after, median [IQR] | Compilation before → after, median [IQR] | Execution before → after, median [IQR] |
| --- | --- | --- | --- |
| Filter | 84.4 [82.3–86.9] → 65.4 [64.9–66.2] | 128.0 [125.0–128.8] → 123.5 [123.1–124.2] | 1045.7 [975.4–1356.3] → 908.6 [907.4–912.2] |
| Log probability | 89.4 [78.5–96.7] → 61.9 [59.3–63.6] | 82.5 [79.7–95.9] → 82.7 [82.5–83.2] | 954.9 [747.1–1157.0] → 632.5 [630.4–646.1] |
| Value and gradient | 194.5 [194.5–198.3] → 162.8 [160.9–163.3] | 221.8 [211.1–234.6] → 212.5 [207.3–216.2] | 2802.3 [2507.1–3097.2] → 2009.1 [2007.4–2032.2] |

The larger baseline's execution timings were variable, so these separate-process
measurements do not establish a general runtime speedup. The likelihood scan's
calculated output storage drops from about 378 MiB to 129 KiB. Compiler-reported
gradient temporary storage remains about 2.25 GB; eliminating filter history does
not eliminate reverse-mode storage. Full filter output times now retain float64
precision, adding four bytes per timestep relative to the old float32 timestamps.

Reproduce a comparison by setting `PYTHONPATH` to each checkout's `src` in separate
processes, running the benchmark with distinct `--output` prefixes, and passing
the baseline `.npz` as `--reference` for the candidate run. JSON includes compiler
argument, output, temporary, and alias memory estimates; `.hlo` files permit
inspection of the compiled loops. These estimates do not include every runtime
or library allocation and are not measurements of peak resident memory.
