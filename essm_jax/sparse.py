from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class SparseRepresentation(NamedTuple):
    shape: Tuple[int, ...]
    rows: jax.Array
    cols: jax.Array
    vals: jax.Array


def create_sparse_rep(m: np.ndarray) -> SparseRepresentation:
    """
    Creates a sparse rep from matrix m. Use in linear models with materialise_jacobian=False for 2x speed up.

    Args:
        m: [N,M] matrix

    Returns:
        sparse rep
    """
    rows, cols = np.where(m)
    sort_indices = np.lexsort((cols, rows))
    rows = rows[sort_indices]
    cols = cols[sort_indices]
    return SparseRepresentation(
        shape=np.shape(m),
        rows=jnp.asarray(rows),
        cols=jnp.asarray(cols),
        vals=jnp.asarray(m[rows, cols])
    )


def to_dense(m: SparseRepresentation, out: jax.Array | None = None) -> jax.Array:
    """
    Form dense matrix.

    Args:
        m: sparse rep
        out: output buffer

    Returns:
        out + M
    """
    if out is None:
        out = jnp.zeros(m.shape, m.vals.dtype)

    return out.at[m.rows, m.cols].add(m.vals, unique_indices=True, indices_are_sorted=True)


def matvec_sparse(m: SparseRepresentation, v: jax.Array, out: jax.Array | None = None) -> jax.Array:
    """
    Compute matmul for sparse rep. Speeds up large sparse linear models by about 2x.

    Args:
        m: sparse rep
        v: vec
        out: output buffer to add to.

    Returns:
        out + M @ v
    """
    if out is None:
        out = jnp.zeros(m.shape[0])
    return out.at[m.rows].add(m.vals * v[m.cols], unique_indices=True, indices_are_sorted=True)
