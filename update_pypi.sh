#!/bin/bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--upload" ) ]]; then
    printf 'Usage: %s [--upload]\n' "$0" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
release_dir=$(mktemp -d "${TMPDIR:-/tmp}/essm-jax-dist.XXXXXX")

conda run -n essm_py python -m build --outdir "$release_dir"
conda run -n essm_py python -m twine check --strict "$release_dir"/*
printf 'Checked distributions: %s\n' "$release_dir"

if [[ "${1:-}" == "--upload" ]]; then
    conda run -n essm_py python -m twine upload "$release_dir"/*
fi
