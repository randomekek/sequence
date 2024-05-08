# %%
import inspect
import jax
import jax.numpy as jnp
import jax.random as jr

# Keep it elementary: jax.Array, list[model]
# Use params and code flow: seq, linear, affine, residual
# Use functions: layer norm, dropout
# Use model: when you need list[model]

# TODO: random & list init, dropout (random key, inference)


def model(fn):
    signature = inspect.signature(fn)
    field_iter = iter(signature.parameters.items())
    first_field = next(field_iter)
    field_names = [k for k, v in field_iter]
    def init(self, **kwargs):
        for f in field_names:
            setattr(self, f, kwargs.get(f))
    def validate(self):
        missing = [f for f, v in vars(self).items() if v is None]
        if missing:
            raise TypeError(f'{fn.__name__} missing field: {" ".join(missing)}')
    def call(self, x):
        validate(self)
        return fn(x, **vars(self))
    def repr(self):
        return fn.__name__ + str(vars(self))
    cls = type(fn.__name__, (), {"__init__": init, "__call__": call, "__repr__": repr})
    def flatten(self):
        return (tuple(getattr(self, k) for k in field_names), None)
    def unflatten(_, items):
        return cls(**{k: v for k, v in zip(field_names, items)})
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


def partition_arrays(t):
    left = jax.tree_util.tree_map(lambda x: x if isinstance(x, jax.Array) else None, t)
    right = jax.tree_util.tree_map(lambda x: None if isinstance(x, jax.Array) else x, t)
    return left, right


def unpartition_arrays(left, right):
    jax.tree_util.tree_map(lambda l, r: l or r, left, right)


def dropout(x, key, p):
    q = 1 - jax.lax.stop_gradient(p)
    mask = jr.bernoulli(key, q, x.shape)
    return jnp.where(mask, x / q, 0)


def layer_norm(x, epsilon):
    mean = jnp.mean(x, keepdims=True)
    variance = jnp.var(x, keepdims=True)
    variance = jnp.maximum(0.0, variance)
    inv = jax.lax.rsqrt(variance + epsilon)
    return (x - mean) * inv
