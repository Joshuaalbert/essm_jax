from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from essm_jax.jvp_op import JVPLinearOp


@pytest.mark.parametrize('linearize', [True, False])
def test_jvp_linear_op(linearize: bool):
    n = 4
    k = 10
    m = 2

    def fn(x):
        return jnp.asarray([jnp.sum(jnp.sin(x) ** i) for i in range(m)])

    x = jnp.arange(n).astype(jnp.float32)

    jvp_op = JVPLinearOp(fn, linearize=linearize)
    jvp_op = jvp_op(x)

    x_space = jnp.ones((n, k))
    f_space = jnp.ones((m, k))

    assert jvp_op.matvec(x_space[:, 0]).shape == (m,)
    assert jnp.allclose(jvp_op.matvec(x_space[:, 0]), jvp_op.to_dense() @ x_space[:, 0])
    assert jnp.allclose(jvp_op @ x_space[:, 0], jvp_op.to_dense() @ x_space[:, 0])

    assert jvp_op.matvec(f_space[:, 0], adjoint=True).shape == (n,)
    assert jnp.allclose(jvp_op.matvec(f_space[:, 0], adjoint=True), jvp_op.to_dense().T @ f_space[:, 0])
    assert jnp.allclose(jvp_op.T @ f_space[:, 0], jvp_op.to_dense().T @ f_space[:, 0])

    assert jvp_op.matmul(x_space).shape == (m, k)
    assert jnp.allclose(jvp_op.matmul(x_space), jvp_op.to_dense() @ x_space)
    assert jnp.allclose(jvp_op @ x_space, jvp_op.to_dense() @ x_space)

    assert jvp_op.matmul(x_space.T, left_multiply=False).shape == (k, m)
    assert jnp.allclose(jvp_op.matmul(x_space.T, left_multiply=False), x_space.T @ jvp_op.to_dense().T)
    assert jnp.allclose(x_space.T @ jvp_op.T, x_space.T @ jvp_op.to_dense().T)

    assert jvp_op.matmul(f_space, adjoint=True).shape == (n, k)
    assert jnp.allclose(jvp_op.matmul(f_space, adjoint=True), jvp_op.to_dense().T @ f_space)
    assert jnp.allclose(jvp_op.T @ f_space, jvp_op.to_dense().T @ f_space)

    assert jvp_op.matmul(f_space.T, adjoint=True, left_multiply=False).shape == (k, n)
    assert jnp.allclose(jvp_op.matmul(f_space.T, adjoint=True, left_multiply=False), f_space.T @ jvp_op.to_dense())
    assert jnp.allclose(f_space.T @ jvp_op, f_space.T @ jvp_op.to_dense())

    # test setting primals
    assert jnp.allclose(jvp_op(x).matvec(x_space[:, 0]), jvp_op.matvec(x_space[:, 0]))

    # Test neg
    assert jnp.allclose((-jvp_op).to_dense(), -jvp_op.to_dense())

    # Test when f: R^n -> scalar
    def fn(x):
        return jnp.sum(jnp.sin(x))

    jvp_op = JVPLinearOp(fn, primals=x, linearize=linearize)
    assert jvp_op.matvec(x_space[:, 0]).shape == ()
    assert jnp.allclose(jvp_op.matvec(x_space[:, 0]), jvp_op.to_dense() @ x_space[:, 0])
    assert jnp.allclose(jvp_op @ x_space[:, 0], jvp_op.to_dense() @ x_space[:, 0])


@pytest.mark.parametrize('init_primals', [True, False])
@pytest.mark.parametrize('linearize', [True, False])
def test_multiple_primals(init_primals: bool, linearize: bool):
    n = 5
    k = 3

    # Test multiple primals
    def fn(x, y):
        return jnp.stack([x * y, y, -y], axis=-1)  # [n, 3]

    x = jnp.arange(n).astype(jnp.float32)
    y = jnp.arange(n).astype(jnp.float32)
    if init_primals:
        jvp_op = JVPLinearOp(fn, primals=(x, y), linearize=linearize)
    else:
        jvp_op = JVPLinearOp(fn, linearize=linearize)
        jvp_op = jvp_op(x, y)
    x_space = jnp.ones((n, k))
    y_space = jnp.ones((n, k))
    assert jvp_op.matvec(x_space[:, 0], y_space[:, 0]).shape == (n, 3)
    assert jvp_op.matmul(x_space, y_space).shape == (n, 3, k)

    with pytest.raises(ValueError, match='Dunder methods currently only defined for operation on arrays.'):
        _ = (jvp_op @ (x_space[:, 0], y_space[:, 0])).shape == (n, 3)
    with pytest.raises(ValueError, match='Dunder methods currently only defined for operation on arrays.'):
        _ = (jvp_op @ (x_space, y_space)).shape == (n, 3, k)


@pytest.mark.parametrize('linearize', [True, False])
def test_jvp_op_dtype_promotion(linearize: bool):
    def fn(x, y):
        return x + y + 0j

    jvp_op = JVPLinearOp(fn, promote_dtypes=True, linearize=linearize)

    primals = (jnp.ones(1), jnp.ones(1))
    jvp_op = jvp_op(*primals)
    np.testing.assert_allclose(jvp_op.matvec(jnp.ones(1) + 0j, jnp.ones(1) + 0j),
                               jvp_op.matvec(jnp.ones(1), jnp.ones(1)))

    np.testing.assert_allclose(jvp_op.matvec(fn(*primals).astype(jnp.float32), adjoint=True),
                               jvp_op.matvec(fn(*primals), adjoint=True))


@pytest.mark.parametrize('linearize', [True, False])
def test_jvp_op_pytree_primals_and_cotangents(linearize: bool):
    class Primal(NamedTuple):
        x: jax.Array
        y: jax.Array

    class Cotangent(NamedTuple):
        x: jax.Array
        y: jax.Array
        z: jax.Array

    class Cotangent2(NamedTuple):
        x: jax.Array
        y: jax.Array
        z: jax.Array
        h: jax.Array

    def f(x: Primal) -> tuple[Cotangent, Cotangent2]:
        return Cotangent(x=x.x, y=x.y, z=x.x + x.y), Cotangent2(x=x.x, y=x.y, z=x.x + x.y, h=x.x - x.y)

    F = JVPLinearOp(f, linearize=linearize)
    primal = Primal(x=jnp.ones(2), y=jnp.ones(2))
    F = F(primal)
    cotangent = Cotangent(jnp.ones(2), jnp.ones(2), jnp.ones(2)), Cotangent2(jnp.ones(2), jnp.ones(2), jnp.ones(2),
                                                                             jnp.ones(2))
    tangent = Primal(jnp.ones(2), jnp.ones(2))

    print(F.matvec(tangent))
    print(F.matvec(*cotangent, adjoint=True))

    def f(x: Primal, y: Primal) -> tuple[Cotangent, Cotangent2]:
        return Cotangent(x=x.x, y=x.y, z=x.x + x.y + y.y), Cotangent2(x=x.x + y.x, y=x.y, z=x.x + x.y, h=x.x - x.y)

    F = JVPLinearOp(f, linearize=linearize)
    primal = Primal(x=jnp.ones(2), y=jnp.ones(2))
    F = F(primal, primal)
    cotangent = Cotangent(jnp.ones(2), jnp.ones(2), jnp.ones(2)), Cotangent2(jnp.ones(2), jnp.ones(2), jnp.ones(2),
                                                                             jnp.ones(2))
    tangent = Primal(jnp.ones(2), jnp.ones(2))

    print(F.matvec(tangent, tangent))
    print(F.matvec(*cotangent, adjoint=True))


def test_linearize():
    def f(x):
        return jnp.sum(jnp.sin(x) ** 2) * jnp.cos(x)

    x = jax.random.normal(jax.random.PRNGKey(0), (10,))
    y = f(x)
    g = JVPLinearOp(f, primals=x, linearize=False)
    g_linear = JVPLinearOp(f, primals=x, linearize=True)

    tangent = jax.random.normal(jax.random.PRNGKey(0), x.shape, x.dtype)
    cotangent = jax.random.normal(jax.random.PRNGKey(0), y.shape, y.dtype)

    jvp = g.matvec(tangent)
    jvp_linear = g_linear.matvec(tangent)

    print(jvp)
    assert jnp.allclose(jvp, jvp_linear)

    vjp = g.matvec(cotangent, adjoint=True)
    vjp_linear = g_linear.matvec(cotangent, adjoint=True)

    print(vjp)
    assert jnp.allclose(vjp, vjp_linear)
