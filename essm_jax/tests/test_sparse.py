import jax
import numpy as np
from jax import numpy as jnp

jax.config.update('jax_enable_x64', True)

from essm_jax.sparse import create_sparse_rep, matvec_sparse, to_dense


def test_sparse_rep():
    m = np.asarray([[1., 0, 0],
                    [-1., 2., 0.],
                    [0., 0., 5.]])
    rep = create_sparse_rep(m)
    v = jnp.asarray([1, 1, 1])
    np.testing.assert_allclose(matvec_sparse(rep, v), m @ v)

    m = np.random.normal(size=(100, 100))
    v = np.random.normal(size=100)

    rep = create_sparse_rep(m)
    np.testing.assert_allclose(matvec_sparse(rep, v), m @ v)


def test_to_dense():
    m = np.asarray([[1., 0, 0],
                    [-1., 2., 0.],
                    [0., 0., 5.]])
    rep = create_sparse_rep(m)
    M = to_dense(rep)
    np.testing.assert_allclose(M, m)
