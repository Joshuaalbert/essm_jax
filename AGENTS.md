# Agent Notes (essm_jax)

- Read `INVARIANTS.md` and `LEARNINGS.md` before changing code.

## Repo Layout

- Package: `src/essm_jax/`.
- Tests: `src/essm_jax/tests/`; name tests `test_*.py`.
- Packaging: `setuptools`, configured entirely in `pyproject.toml`.
- Python: `>=3.10`; CI covers Python 3.10 through 3.14 on CPU.
- Dependencies: `[project.dependencies]` in `pyproject.toml`; the `tests` and
  `examples` extras are in `[project.optional-dependencies]`.
- CI: `.github/workflows/unittests.yml`; release tooling: `update_pypi.sh`.

## Run Things

Use the `essm_py` conda environment for all local Python commands. Conda manages
the interpreter; install dependencies with pip inside that environment.

```bash
conda run -n essm_py pip install -e '.[tests]'
conda run -n essm_py pytest
conda run -n essm_py pytest --cov --cov-config=pyproject.toml --cov-report=term-missing --cov-report=xml --cov-report=html
conda run -n essm_py python -m flake8 src --select=E9,F63,F7,F82 --show-source --statistics
conda run -n essm_py ruff check src
conda run -n essm_py python -m build
conda run -n essm_py python -m twine check --strict dist/*
```

For terminal snippets, likewise use `conda run -n essm_py python -c '...'`.
The timing benchmark is opt-in: `conda run -n essm_py pytest -m benchmark`.

## Code Style Guidelines

- Imports: stdlib, third-party, local; prefer absolute package imports; no `*`.
- Formatting: four spaces, no tabs, readable line wrapping, trailing commas in
  multi-line expressions, and f-strings. Follow the surrounding code.
- Types: use Python 3.10-compatible annotations; type public APIs and non-trivial
  internals; avoid `Any` where a meaningful type is available.
- Naming: modules/functions/variables use `snake_case`, classes use `PascalCase`,
  constants use `UPPER_SNAKE_CASE`, and private names start with `_`.
- Keep production code in `src/essm_jax/` and tests beside the package in its
  `tests/` directory. Keep `__init__.py` files empty.
- Errors: fail fast with specific exceptions; no bare `except:`; add context
  with `raise ... from e` when appropriate.
- Logging: use `logging.getLogger(__name__)` if logging is needed. Avoid ad-hoc
  global logging configuration and logging in numerical hot loops.
- Performance: keep code clear first and optimize with evidence. Compare
  optimized numerical paths with a simple dense/reference calculation.
- Refactor duplicated code only when it is non-trivial (more than five lines)
  and the abstraction makes the algorithm easier to follow. Prefer linear
  readability to helpers that add mental bookkeeping.
- Benchmarks and reporting scripts should read in execution order: setup, run,
  collect results, write report. Factor helpers only when meaningful repeated
  work benefits from a clear name.
- Track array shapes where non-obvious arrays are created or transformed, such
  as `# [num_time, latent_size]` and `# [latent_size, latent_size]`.
- Use Google-style API docstrings with concise Args, Returns, and Raises
  sections where useful. Keep time conventions, units, and scales explicit.
- Preserve the existing NamedTuple result/state containers and their JAX pytree
  behavior. Do not add a dependency on a pytree base class from another project.
- Tests: use deterministic inputs and explicit PRNG keys; isolate timing and
  external services. Use parametrization for numerical variants and NamedTuples
  for fixtures that need a schema; avoid fixtures for trivial test data.
- Keep all dependency declarations in `pyproject.toml`: runtime dependencies in
  `project.dependencies`, and test/lint/build tools in the `tests` extra. Keep
  the declared package extras working; do not reintroduce requirements files or
  `setup.py`.
- No formatter or type checker is enforced. Ruff and flake8 gate syntax and
  undefined-name errors; CI reports other flake8 style findings without failing.

## Agent Workflow Expectations

- Preserve `INVARIANTS.md`; update the document and regression tests together
  when intentionally changing a contract.
- Add or adjust tests for behavior changes. Run the relevant tests, flake8, and
  Ruff; run the full coverage suite for changes to numerical code or CI.
- Coverage includes every production module, including unexecuted modules, and
  excludes tests. Do not reduce the 80% line-and-branch coverage gate or omit
  production code to make a check pass.
- Separate compilation and steady-state execution when reporting JAX timings;
  synchronize results before recording elapsed time.
- Keep changes scoped to the task. Do not add repo-wide tooling unless asked.
- Keep modules small and public imports usable from `essm_jax...`.
- Record verified, reusable project discoveries in `LEARNINGS.md`.

## Repo Hygiene

- Do not commit credentials, environments, caches, build artifacts, or generated
  coverage/test reports.
- Do not add editor-specific settings unless requested.
- Package publication is explicit and manual; building a distribution is not
  authorization to upload it.
