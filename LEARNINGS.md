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
- The public helper module is named `pytee_utils.py` (historical spelling).
  Preserve that import path when doing unrelated work.
