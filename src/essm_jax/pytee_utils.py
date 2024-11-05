from typing import Tuple, Callable, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

PT = TypeVar('PT')


def pytree_unravel(example_tree: PT) -> Tuple[Callable[[PT], jax.Array], Callable[[jax.Array], PT]]:
    """
    Returns functions to ravel and unravel a pytree.

    Args:
        example_tree: a pytree to be unravelled, can also be a pytree of ShapeDtypeStruct objects instead of arrays.

    Returns:
        ravel_fun: a function to ravel the pytree
        unravel_fun: a function to unravel
    """
    leaf_list, tree_def = jax.tree.flatten(example_tree)

    sizes = [np.size(leaf) for leaf in leaf_list]
    shapes = [np.shape(leaf) for leaf in leaf_list]
    dtypes = [leaf.dtype for leaf in leaf_list]

    def ravel_fun(pytree: PT) -> jax.Array:
        leaf_list, tree_def = jax.tree.flatten(pytree)
        # promote types to common one
        common_dtype = jnp.result_type(*dtypes)
        leaf_list = [leaf.astype(common_dtype) for leaf in leaf_list]
        return jnp.concatenate([lax.reshape(leaf, (size,)) for leaf, size in zip(leaf_list, sizes)])

    def unravel_fun(flat_array: jax.Array) -> PT:
        leaf_list = []
        start = 0
        for size, shape, dtype in zip(sizes, shapes, dtypes):
            leaf_list.append(lax.reshape(flat_array[start:start + size], shape).astype(dtype))
            start += size
        return jax.tree.unflatten(tree_def, leaf_list)

    return ravel_fun, unravel_fun
