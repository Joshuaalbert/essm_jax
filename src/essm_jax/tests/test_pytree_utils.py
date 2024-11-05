from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp

from essm_jax.pytee_utils import pytree_unravel


def test_pytree_unravel():
    # Simple test
    example_tree = {'a': np.array([1, 2, 3]), 'b': np.array([[4, 5], [6, 7]])}
    ravel_fun, unravel_fun = pytree_unravel(example_tree)
    flat = ravel_fun(example_tree)
    assert jnp.allclose(flat, jnp.array([1, 2, 3, 4, 5, 6, 7]))
    assert isinstance(unravel_fun(flat), dict)
    assert jnp.allclose(unravel_fun(flat)['a'], example_tree['a'])
    assert jnp.allclose(unravel_fun(flat)['b'], example_tree['b'])

    # with named tuple
    class State(NamedTuple):
        a: np.ndarray
        b: np.ndarray

    example_tree = State(np.array([1, 2, 3]), np.array([[4, 5], [6, 7]]))
    ravel_fun, unravel_fun = pytree_unravel(example_tree)
    flat = ravel_fun(example_tree)
    assert jnp.allclose(flat, jnp.array([1, 2, 3, 4, 5, 6, 7]))
    assert isinstance(unravel_fun(flat), State)
    assert jnp.allclose(unravel_fun(flat).a, example_tree.a)
    assert jnp.allclose(unravel_fun(flat).b, example_tree.b)

    # with ShapeDtype
    example_tree_def = dict(
        a=jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
        b=jax.ShapeDtypeStruct(shape=(3, 2), dtype=jnp.float32)
    )
    example_tree = dict(
        a=np.ones((2, 3), dtype=jnp.float32),
        b=np.ones((3, 2), dtype=jnp.float32)
    )
    ravel_fun, unravel_fun = pytree_unravel(example_tree_def)
    flat = ravel_fun(example_tree)
    assert jnp.allclose(flat, jnp.ones(12))
    assert isinstance(unravel_fun(flat), dict)
    assert jnp.allclose(unravel_fun(flat)['a'], example_tree['a'])
    assert jnp.allclose(unravel_fun(flat)['b'], example_tree['b'])
