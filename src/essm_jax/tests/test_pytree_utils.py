from typing import NamedTuple

import jax
import numpy as np
import pytest
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


def test_pytree_unravel_preserves_mixed_leaf_dtypes_under_jit():
    example = {
        "counts": jnp.array([1, 3], dtype=jnp.int16),
        "values": jnp.array([[0.5], [2.5]], dtype=jnp.float32),
    }
    ravel, unravel = pytree_unravel(example)
    restored = jax.jit(lambda tree: unravel(ravel(tree)))(example)
    assert jax.tree.structure(restored) == jax.tree.structure(example)
    for key, expected in example.items():
        assert restored[key].dtype == expected.dtype
        assert restored[key].shape == expected.shape
        np.testing.assert_array_equal(restored[key], expected)


@pytest.mark.parametrize('invalid', [
    {'a': jnp.ones((2, 2)), 'b': jnp.ones(1)},
    {'b': jnp.ones((2, 2))},
    {'a': jnp.ones(4)},
])
def test_ravel_rejects_incompatible_structure_and_shapes(invalid):
    ravel, _ = pytree_unravel({'a': jnp.ones((2, 2))})
    with pytest.raises(ValueError, match='structure|shape'):
        jax.jit(ravel)(invalid)


@pytest.mark.parametrize('shape', [(3,), (5,), (2, 2)])
def test_unravel_rejects_wrong_flat_shape(shape):
    _, unravel = pytree_unravel({'a': jnp.ones((2, 2))})
    with pytest.raises(ValueError, match='shape'):
        jax.jit(unravel)(jnp.ones(shape))


@pytest.mark.parametrize('empty', [{}, (), {'a': None}])
def test_empty_pytree_round_trip(empty):
    ravel, unravel = pytree_unravel(empty)
    flat = jax.jit(ravel)(empty)
    assert flat.shape == (0,)
    assert flat.dtype == jnp.float32
    assert jax.jit(unravel)(flat) == empty
