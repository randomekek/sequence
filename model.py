# %%
import inspect
import jax
import jax.numpy as jnp

# Keep it elementary: jax.Array, list[model]
# Use params and code flow: seq, linear, affine, residual
# Use functions: layer norm, dropout
# Use model: when you need list[model]

# TODO: random & list init, dropout (random key, inference)


def model(fn):
    compiled = jax.jit(fn)
    signature = inspect.signature(fn)
    field_iter = iter(signature.parameters.items())
    first_field = next(field_iter)
    field_names = [k for k, v in field_iter]
    def init(self, **kwargs):
        for f in field_names:
            setattr(self, f, kwargs.get(f))
    def validate(kwargs):
        missing = [f for f, v in kwargs.items() if v is None]
        if missing:
            raise TypeError(f'{fn.__name__} missing field: {" ".join(missing)}')
    def call(self, x, **kwargs):
        kwargs = vars(self) | kwargs
        validate(kwargs)
        return compiled(x, **kwargs)
    def repr(self):
        return fn.__name__ + str(vars(self))
    cls = type(fn.__name__, (), {"__init__": init, "__call__": call, "__repr__": repr})
    def flatten(self):
        return ((k, getattr(self, k)) for k in field_names), None
    def flatten_fast(self):
        return (getattr(self, k) for k in field_names), None
    def unflatten(_, items):
        return cls(**{k: v for k, v in zip(field_names, items)})
    jax.tree_util.register_pytree_with_keys(cls, flatten, unflatten, flatten_fast)
    return cls


class Initializer(object):
    def __init__(self, key):
        self.key = key

    def normal(self, shape, stddev):
        self.key, key = jax.random.split(self.key)
        return jax.nn.initializers.normal(stddev)(key, shape)

    def golorot_normal(self, shape):
        self.key, key = jax.random.split(self.key)
        return jax.nn.initializers.glorot_normal()(key, shape)

    def map(self, num, fn):
        self.key, *keys = jax.random.split(self.key, num + 1)
        return [fn(Initializer(x)) for x in keys]


def partition_arrays(t):
    left = jax.tree_util.tree_map(lambda x: x if isinstance(x, jax.Array) else None, t)
    right = jax.tree_util.tree_map(lambda x: None if isinstance(x, jax.Array) else x, t)
    return left, right


def unpartition(left, right):
    merge = lambda l, r: l or r
    is_none = lambda x: x is None
    return jax.tree_util.tree_map(merge, left, right, is_leaf=is_none)


def dropout(x, key, p):
    q = 1 - jax.lax.stop_gradient(p)
    mask = jax.random.bernoulli(key, q, x.shape)
    return jnp.where(mask, x / q, 0)


def layer_norm(x, epsilon):
    mean = jnp.mean(x, keepdims=True)
    variance = jnp.var(x, keepdims=True)
    variance = jnp.maximum(0.0, variance)
    inv = jax.lax.rsqrt(variance + epsilon)
    return (x - mean) * inv
