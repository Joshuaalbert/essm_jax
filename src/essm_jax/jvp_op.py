import dataclasses
import inspect
import os
import warnings
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np


def get_grandparent_info(relative_depth: int = 7):
    """
    Get the file, line number and function name of the caller of the caller of this function.

    Args:
        relative_depth: the number of frames to go back from the caller of this function. Default is 6. Should be
        enough to get out of a jax.tree.map call.

    Returns:
        str: a string with the file, line number and function name of the caller of the caller of this function.
    """
    # Get the grandparent frame (caller of the caller)
    s = []
    for depth in range(1, min(1 + relative_depth, len(inspect.stack()) - 1) + 1):
        caller_frame = inspect.stack()[depth]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        s.append(f"{os.path.basename(caller_file)}:{caller_line} in {caller_func}")
    s = s[::-1]
    s = f"at {' -> '.join(s)}"
    return s


def isinstance_namedtuple(obj) -> bool:
    """
    Check if object is a namedtuple.

    Args:
        obj: object

    Returns:
        bool
    """
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


def make_linear(f: Callable, *primals0):
    """
    Make a linear function that approximates f around primals0.

    Args:
        f: the function to linearize
        *primals0: the point around which to linearize

    Returns:
        the linearized function
    """
    f0, f_jvp = jax.linearize(f, *primals0)

    def f_linear(*primals):
        diff_primals = jax.tree.map(lambda x, x0: x - x0, primals, primals0)
        df = f_jvp(*diff_primals)
        return jax.tree.map(lambda y0, dy: y0 + dy, f0, df)

    return f_linear


@dataclasses.dataclass(eq=False)
class JVPLinearOp:
    """
    Represents J_ij = d/d x_j f_i(x), where x is the primal value.

    This is a linear operator that represents the Jacobian of a function.
    """
    fn: Callable  # A function R^n -> R^m
    primals: Any | None = None  # The primal value, i.e. where jacobian is evaluated
    more_outputs_than_inputs: bool = False  # If True, the operator is tall, i.e. m > n
    adjoint: bool = False  # If True, the operator is transposed
    promote_dtypes: bool = True  # If True, promote dtypes to match primal during JVP, and cotangent to match primal_out during VJP
    linearize: bool = True  # If True, use linearized function for JVP

    def __post_init__(self):
        if not callable(self.fn):
            raise ValueError('`fn` must be a callable.')

        if self.primals is not None:
            if isinstance_namedtuple(self.primals) or (not isinstance(self.primals, tuple)):
                self.primals = (self.primals,)
            if self.linearize:
                self.linear_fn = make_linear(self.fn, *self.primals)

    def __call__(self, *primals: Any) -> 'JVPLinearOp':
        return JVPLinearOp(
            fn=self.fn,
            primals=primals,
            more_outputs_than_inputs=self.more_outputs_than_inputs,
            adjoint=self.adjoint,
            promote_dtypes=self.promote_dtypes,
            linearize=self.linearize
        )

    def __neg__(self):
        return JVPLinearOp(
            fn=lambda *args, **kwargs: jax.lax.neg(self.fn(*args, **kwargs)),
            primals=self.primals,
            more_outputs_than_inputs=self.more_outputs_than_inputs,
            adjoint=self.adjoint,
            promote_dtypes=self.promote_dtypes,
            linearize=self.linearize
        )

    def __matmul__(self, other):
        if not isinstance(other, (jax.Array, np.ndarray)):
            raise ValueError(
                'Dunder methods currently only defined for operation on arrays. '
                'Use .matmul(...) for general tangents.'
            )
        if len(np.shape(other)) == 1:
            return self.matvec(other, adjoint=self.adjoint)
        return self.matmul(other, adjoint=self.adjoint, left_multiply=True)

    def __rmatmul__(self, other):
        if not isinstance(other, (jax.Array, np.ndarray)):
            raise ValueError(
                'Dunder methods currently only defined for operation on arrays. '
                'Use .matmul(..., left_multiply=False) for general tangents.'
            )
        if len(np.shape(other)) == 1:
            return self.matvec(other, adjoint=not self.adjoint)
        return self.matmul(other, adjoint=not self.adjoint, left_multiply=False)

    @property
    def T(self) -> 'JVPLinearOp':
        return JVPLinearOp(
            fn=self.fn,
            primals=self.primals,
            more_outputs_than_inputs=self.more_outputs_than_inputs,
            adjoint=not self.adjoint,
            promote_dtypes=self.promote_dtypes,
            linearize=self.linearize
        )

    def matmul(self, *tangents: Any, adjoint: bool = False, left_multiply: bool = True):
        """
        Implements matrix multiplication from matvec using vmap.

        Args:
            tangents: pytree of the same structure as the primals, but with appropriate more columns for adjoint=False,
                or more rows for adjoint=True.
            adjoint: if True, compute J.T @ v, else compute J @ v
            left_multiply: if True, compute M @ J, else compute J @ M

        Returns:
            pytree of matching either f-space (output) or x-space (primals)
        """
        if left_multiply:
            # J.T @ M or J @ M
            in_axes = -1
            out_axes = -1
        else:
            # M @ J.T or M @ J
            in_axes = 0
            out_axes = 0
        if adjoint:
            return jax.vmap(lambda *_tangent: self.matvec(*_tangent, adjoint=adjoint),
                            in_axes=in_axes, out_axes=out_axes)(*tangents)
        return jax.vmap(lambda *_tangent: self.matvec(*_tangent, adjoint=adjoint),
                        in_axes=in_axes, out_axes=out_axes)(*tangents)

    def matvec(self, *tangents: Any, adjoint: bool = False):
        """
        Compute J @ v = sum_j(J_ij * v_j) using a JVP, if adjoint is False.
        Compute J.T @ v = sum_i(v_i * J_ij) using a VJP, if adjoint is True.

        Args:
            tangents: if adjoint=False, then  pytree of the same structure as the primals, else pytree of the same
                structure as the output.
            adjoint: if True, compute J.T @ v, else compute J @ v

        Returns:
            pytree of matching either f-space (output) if adjoint=False, else x-space (primals)
        """
        if self.primals is None:
            raise ValueError("The primal value must be set to compute the Jacobian.")

        if adjoint:
            co_tangents = tangents

            def _get_results_type(primal_out: jax.Array):
                return primal_out.dtype

            def _adjoint_promote_dtypes(co_tangent: jax.Array, dtype: jnp.dtype):
                if co_tangent.dtype != dtype:
                    warnings.warn(
                        f"Promoting co-tangent dtype from {co_tangent.dtype} to {dtype}, {get_grandparent_info()}."
                    )
                return co_tangent.astype(dtype)

            # v @ J
            if self.linearize:
                f_vjp = jax.linear_transpose(self.linear_fn, *self.primals)
                primals_out = jax.eval_shape(self.linear_fn, *self.primals)
            else:
                primals_out, f_vjp = jax.vjp(self.fn, *self.primals)

            if isinstance_namedtuple(primals_out) or (not isinstance(primals_out, tuple)):
                # JAX squeezed structure to a single element, as the function only returns one output
                co_tangents = co_tangents[0]

            if self.promote_dtypes:
                result_type = jax.tree.map(_get_results_type, primals_out)
                co_tangents = jax.tree.map(_adjoint_promote_dtypes, co_tangents, result_type)

            del primals_out
            output = f_vjp(co_tangents)
            if len(output) == 1:
                return output[0]
            return output

        def _promote_dtype(primal: jax.Array, dtype: jnp.dtype):
            if primal.dtype != dtype:
                warnings.warn(f"Promoting primal dtype from {primal.dtype} to {dtype}, at {get_grandparent_info()}.")
            return primal.astype(dtype)

        def _get_result_type(primal: jax.Array):
            return primal.dtype

        primals = self.primals
        if self.promote_dtypes:
            result_types = jax.tree.map(_get_result_type, primals)
            tangents = jax.tree.map(_promote_dtype, tangents, result_types)
        # We use linearised function, so that repeated applications are cheaper.
        if self.linearize:
            primal_out, tangent_out = jax.jvp(self.linear_fn, primals, tangents)
        else:
            primal_out, tangent_out = jax.jvp(self.fn, primals, tangents)
        return tangent_out

    def to_dense(self) -> jax.Array:
        """
        Compute the dense Jacobian at a point.

        Returns:
            [m, n] array
        """
        if self.primals is None:
            raise ValueError("The primal value must be set to compute the Jacobian.")

        if self.more_outputs_than_inputs:
            return jax.jacfwd(self.fn)(*self.primals)
        return jax.jacrev(self.fn)(*self.primals)
