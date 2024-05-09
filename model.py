# %%
import inspect
import jax
import jax.numpy as jnp

# Use jax.Array: linear, affine, bias
# Use code: seq, residual
# Use parameters: prngkey
# Use functions: layer norm, dropout, init
# Use @model: top level, and list[model]


def model(fn):
    parameters = list(inspect.signature(fn).parameters.items())
    fields = [k for k, v in parameters if v.annotation not in (bool, int, float)]
    static = [k for k, v in parameters if v.annotation in (bool, int, float)]
    all_fields_set = set(fields + static)
    def init(self, **kwargs):
        excess = set(kwargs.keys()) - all_fields_set
        if excess:
            raise TypeError(f'{fn.__name__} does not have parameter: {excess}')
        for f in all_fields_set:
            setattr(self, f, kwargs.get(f))
    def check_arguments(args, kwargs):
        for arg, (k, v) in zip(args, parameters):
            if kwargs[k] is not None:
                raise TypeError(f'{fn.__name__} duplicate parameter: {k}')
            kwargs[k] = arg
        for k, v in kwargs.items():
            if v is None:
                raise TypeError(f'{fn.__name__} missing parameter: {k}')
        return kwargs
    def call(self, *args, **kwargs):
        return fn(**check_arguments(args, vars(self) | kwargs))
    def repr(self):
        out = [fn.__name__ + ' {']
        for k, v in jax.tree_util.tree_flatten_with_path(self)[0]:
            out.append(' {} = {}'.format(jax.tree_util.keystr(k), getattr(v, 'shape', None) or str(v)))
        return '\n'.join(out) + '\n}'
    cls = type(fn.__name__, (), {"__init__": init, "__call__": call, "__repr__": repr})
    def flatten(self):
        return ((jax.tree_util.GetAttrKey(k), getattr(self, k)) for k in fields), (getattr(self, k) for k in static)
    def flatten_fast(self):
        return (getattr(self, k) for k in fields), (getattr(self, k) for k in static)
    def unflatten(static_items, field_items):
        return cls(**{k: v for k, v in zip(fields, field_items)}, **{k: v for k, v in zip(static, static_items)})
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
