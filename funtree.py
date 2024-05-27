import inspect
import jax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey, register_pytree_with_keys


class Unset:
    pass


UNSET = Unset()


register_pytree_with_keys(Unset, lambda x: ((), ()), lambda x, y: Unset())


def makefun(fn):
    name = fn.__name__
    params = list(inspect.signature(fn).parameters.items())
    fields = [k for k, v in params if v.annotation not in (bool, int, float)]
    static = [k for k, v in params if v.annotation in (bool, int, float)]
    all_fields = set(fields + static)
    def init(self, **kwargs):
        for k in set(kwargs.keys()) - all_fields:
            raise TypeError(f'{name}() got an unexpected keyword argument: {k}')
        for k, v in kwargs.items():
            setattr(self, k, v)
    def call(self, *args, **kwargs):
        return fn(*args, **vars(self), **kwargs)
    def rep(self):
        out = [f'{name} {{']
        for k, v in vars(self).items():
            out.append(f'\n {"static " if k in static else ""}{k} = ')
            if isinstance(v, jax.Array):
                out.append(f'{v.dtype}{list(v.shape)}')
            else:
                out.append(repr(v).replace('\n', '\n  '))
        return ''.join(out + ['\n}'])
    cls = type(name, (), {"__init__": init, "__call__": call, "__repr__": rep})
    flat_names = [GetAttrKey(k) for k in fields]
    def pack(self, ks):
        return tuple(getattr(self, k, UNSET) for k in ks)
    def flatten(self):
        return tuple(zip(flat_names, pack(self, fields))), pack(self, static)
    def flatten_fast(self):
        return pack(self, fields), pack(self, static)
    def unpack(ks, vs):
        return {k: v for k, v in zip(ks, vs) if not isinstance(v, Unset)}
    def unflatten(svs, fvs):
        return cls(**unpack(static, svs), **unpack(fields, fvs))
    register_pytree_with_keys(cls, flatten, unflatten, flatten_fast)
    return cls


def funmap(tree, mapfns):
    cls = type(tree)
    if cls in mapfns.keys():
        return cls(**mapfns[cls](**{k: funmap(v, mapfns) for k, v in vars(tree).items()}))
    elif cls in (list, tuple):
        return cls(funmap(x, mapfns) for x in tree)
    elif cls == dict:
        return {k: funmap(v, mapfns) for k, v in tree.items()}
    else:
        return tree


def _add_init_methods(cls):
    def wrap_fn(method):
        def inner(self, shape, *args, **kwargs):
            self.key, key = jax.random.split(self.key)
            return getattr(jax.nn.initializers, method)(*args, **kwargs)(key, shape)
        return inner
    for m in ('glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'truncated_normal'):
        setattr(cls, m, wrap_fn(m))
    return cls


@_add_init_methods
class Initializer:
    def __init__(self, key):
        self.key = key

    def map(self, fns):
        self.key, *keys = jax.random.split(self.key, len(fns) + 1)
        return [fn(Initializer(key)) for fn, key in zip(fns, keys)]

    def split(self):
        self.key, key = jax.random.split(self.key)
        return key


def dropout(x, key, p):
    q = 1 - jax.lax.stop_gradient(p)
    return jnp.where(jax.random.bernoulli(key, q, x.shape), x / q, 0)


def norm(x, eps=1e-6):
    mean = jnp.mean(x, keepdims=True)
    var = jnp.maximum(jnp.var(x, keepdims=True), 0.0)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def rms_norm(x, eps=1e-6):
    ms = jnp.mean(jnp.square(x), keepdims=True)
    return x * jax.lax.rsqrt(ms + eps)
