# %%
import inspect
import jax

# Keep it elementary: jax.Array, list[model]
# Use params and code flow: seq, linear, affine, residual
# Use functions: layer norm, dropout


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
