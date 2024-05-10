# %%
import inspect
import jax
import jax.numpy as jnp
from jax.tree_util import keystr, register_pytree_with_keys, tree_leaves_with_path

# Use jax.Array: linear, affine, bias
# Use code: seq, residual
# Use parameters: prngkey
# Use functions: layer norm, dropout, init
# Use @model: top level, and list[model]


class Unset(object): pass


register_pytree_with_keys(Unset, lambda x: ((), ()), lambda x, y: Unset())


def model(fn):
    name = fn.__name__
    params = list(inspect.signature(fn).parameters.items())
    fields = [k for k, v in params if v.annotation not in (bool, int, float)]
    static = [k for k, v in params if v.annotation in (bool, int, float)]
    all_fields = fields + static
    def init(self, **kwargs):
        for k in set(kwargs.keys()) - set(all_fields):
            raise TypeError(f'{name}() got an unexpected keyword argument: {k}')
        for k, v in kwargs.items():
            setattr(self, k, v)
    def call(self, *args, **kwargs):
        return fn(*args, **vars(self), **kwargs)
    def repr(self):
        out = [f'{name} {{']
        for k, v in tree_leaves_with_path(self):
            out.append(f'\n {keystr(k)} = ')
            out.append(f'{v.dtype}{list(v.shape)}' if v is jax.Array else str(v))
        return ''.join(out + ['\n}'])
    cls = type(name, (), {"__init__": init, "__call__": call, "__repr__": repr})
    # hacks: static fields are flattened for repr, but are ignored for unflatten
    flat_names = [f'.{k}' for k in fields] + [f'.{k} (static)' for k in static]
    def pack(self, ks):
        return tuple(getattr(self, k, Unset()) for k in ks)
    def flatten(self):
        return tuple(zip(flat_names, pack(self, all_fields))), pack(self, static)
    def flatten_fast(self):
        return pack(self, all_fields), pack(self, static)
    def unpack(ks, vs):
        return {k: v for k, v in zip(ks, vs) if not isinstance(v, Unset)}
    def unflatten(svs, fvs):
        return cls(**unpack(static, svs), **unpack(fields, fvs))
    register_pytree_with_keys(cls, flatten, unflatten, flatten_fast)
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
