import datetime
import inspect
import jax
import jax.numpy as jnp
import msgpack
import pathlib
import sys


class OutputLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.flush_time = datetime.datetime.now()

    def __enter__(self):
        sys.stdout, sys.stderr = self, self

    def __exit__(self, type, value, traceback):
        sys.stdout, sys.stderr = self.stdout, self.stderr

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            getattr(self.log_file, name)(*args, **kwargs)
            if (datetime.datetime.now() - self.flush_time).total_seconds() > 5.0:
                self.log_file.flush()
            return getattr(self.stdout, name)(*args, **kwargs)
        return inner


def run(fn, description, record=True):
    if not record:
        print('NOT SAVING')
        return fn()
    root = pathlib.Path('logs')
    now = lambda fmt: datetime.datetime.now().strftime(fmt)
    folder = root.joinpath(now('%Y%m'))
    folder.mkdir(exist_ok=True)
    code_filename = folder.joinpath(now('%Y%m%d-%H%M%S') + '.py')
    print(f'run logging to: {code_filename}')
    with open(f'log.txt', 'a+') as summary:
        summary.write(f'===\n{code_filename}\n\n{description}\n\n')
    with open(code_filename, 'x') as code_file:
        code_file.write(f'"""\n{description}\n"""\n\n{inspect.getsource(fn)}\n\n"""\n')
        code_file.flush()
        with OutputLogger(code_file):
            outputs = fn()
        code_file.write('"""\n')
    return outputs


def split_shape(key, shape):
    if isinstance(shape, (tuple, list)):
        return tuple(split_shape(k, s) for (k, s) in zip(jax.random.split(key, len(shape)), shape))
    elif shape == 0:
        return key
    else:
        return jax.random.split(key, shape)


CONST_LIST = {}


def run_const(fn):
    code = inspect.getsource(fn)
    if code not in CONST_LIST:
        name = fn.__name__ if fn.__name__ != '<lambda>' else code
        print(f'eval: {name}')
        CONST_LIST[code] = fn()
    return CONST_LIST[code]


def optimize(model, opt_state, update, accuracy_fn, iter_count=2000, seed=0):
    t = datetime.datetime.now()
    displayed = 0
    last = 0
    key_zero = jax.random.PRNGKey(0)
    try:
        for b, key in zipkey(range(iter_count), jax.random.PRNGKey(seed)):
            loss, model, opt_state = update(key, model, opt_state)
            spent = (datetime.datetime.now() - t).total_seconds()
            if b < 1 or spent > 5.0:
                speed = (b - last) / spent
                print(f'{displayed:02d} {b:04d} {float(loss):.3e} {speed:.1f}it/s ', end='')
                print(f'{accuracy_fn(model, key)*100:0.1f}%')
                t = datetime.datetime.now()
                last = b
                displayed += 1
    except KeyboardInterrupt:
        print('interrupt')

    loss, unused_model, unused_opt_state = update(key_zero, model, opt_state)
    print(f'xx {b:04d} {float(loss):.3e} accuracy {accuracy_fn(model, key_zero)*100:0.1f}% (done)')

    return model, opt_state


def param_count(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


def zipkey(items, key):
    return zip(items, jax.random.split(key, len(items)))


def load_pytree(blob, classes=()):
    class_dict = {c.__name__: c for c in [dict, list, tuple] + classes}
    make_pytree = lambda node, childs: jax.tree_util.PyTreeDef.make_from_node_data_and_children(
        jax.tree_util.default_registry, node, childs)
    def unpack_array(x):
        if isinstance(x, dict) and x.get('funtree') == 'jax.Array':
            return jnp.reshape(jnp.frombuffer(x['bytes'], jnp.dtype(x['dtype'])), x['shape'])
        return x
    def unpack_pytree(obj):
        if obj is None:
            return make_pytree(None, [])
        cls = class_dict[obj['name']]
        return make_pytree((cls, obj['datas']), [unpack_pytree(x) for x in obj['childs']])
    obj = msgpack.unpackb(blob)
    leaves = [unpack_array(x) for x in obj['leaves']]
    pytree = unpack_pytree(obj['pytree'])
    return jax.tree_util.tree_unflatten(pytree, leaves)


def save_pytree(tree):
    leaves, pytree = jax.tree_util.tree_flatten(tree)
    def pack_array(x):
        if isinstance(x, jax.Array):
            return dict(funtree='jax.Array', bytes=x.tobytes(), dtype=str(x.dtype), shape=x.shape)
        return x
    def pack_pytree(treedef: jax.tree_util.PyTreeDef):
        data = treedef.node_data()
        if not data:
            return None
        return dict(name=data[0].__name__, datas=data[1], childs=[pack_pytree(x) for x in treedef.children()])
    return msgpack.packb({
        'leaves': [pack_array(x) for x in leaves],
        'pytree': pack_pytree(pytree),
    })


def cast_pytree(tree, dtype):
    def replace(x):
        if isinstance(x, jax.Array):
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(replace, tree)
