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

    The returned functions reject mismatched structures, leaf shapes, and flat
    array shapes, including during JIT tracing. An empty tree ravels to float32[0].
    """
    leaf_list, tree_def = jax.tree.flatten(example_tree)

    sizes = [np.size(leaf) for leaf in leaf_list]
    shapes = [np.shape(leaf) for leaf in leaf_list]
    dtypes = [leaf.dtype for leaf in leaf_list]
    total_size = sum(sizes)
    common_dtype = jnp.result_type(*dtypes) if dtypes else jnp.float32

    def ravel_fun(pytree: PT) -> jax.Array:
        leaf_list, actual_tree_def = jax.tree.flatten(pytree)
        if actual_tree_def != tree_def:
            raise ValueError('pytree structure must match the example tree')
        if [np.shape(leaf) for leaf in leaf_list] != shapes:
            raise ValueError('pytree leaf shapes must match the example tree')
        if not leaf_list:
            return jnp.empty((0,), dtype=common_dtype)
        # promote types to common one
        leaf_list = [leaf.astype(common_dtype) for leaf in leaf_list]
        return jnp.concatenate([lax.reshape(leaf, (size,)) for leaf, size in zip(leaf_list, sizes)])

    def unravel_fun(flat_array: jax.Array) -> PT:
        if np.shape(flat_array) != (total_size,):
            raise ValueError(f'flat array must have shape ({total_size},), got {np.shape(flat_array)}')
        leaf_list = []
        start = 0
        for size, shape, dtype in zip(sizes, shapes, dtypes):
            leaf_list.append(lax.reshape(flat_array[start:start + size], shape).astype(dtype))
            start += size
        return jax.tree.unflatten(tree_def, leaf_list)

    return ravel_fun, unravel_fun
