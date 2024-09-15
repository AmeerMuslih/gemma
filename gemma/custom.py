import numpy as onp
import opt_einsum

def einsum(*operands, **kwargs):
  optimize = kwargs.pop('optimize', 'auto')
  optimize = 'greedy' if optimize is True else optimize
  if kwargs:
    msg = 'invalid keyword arguments for einsum: {}'
    raise TypeError(msg.format(', '.join(kwargs)))
  # using einsum_call=True here is an internal api for opt_einsum
  operands, contractions = opt_einsum.contract_path(
      *operands, einsum_call=True, use_blas=True, optimize=optimize)
  contractions = tuple(data[:3] for data in contractions)
  return _einsum(operands, contractions)

@_wraps(onp.einsum_path)
def einsum_path(subscripts, *operands, **kwargs):
  optimize = kwargs.pop('optimize', 'greedy')
  # using einsum_call=True here is an internal api for opt_einsum
  return opt_einsum.contract_path(subscripts, *operands, optimize=optimize)

@partial(jit, static_argnums=(1,))
def _einsum(operands, contractions):
  operands = list(_promote_dtypes(*operands))
  sum = lambda x, axes: lax.reduce(x, onp.array(0, x.dtype), lax.add, axes)