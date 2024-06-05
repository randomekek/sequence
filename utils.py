import datetime
import inspect
import jax
import jax.numpy as jnp
import msgpack
import os
import pathlib
import sys

SCALES = [0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.]


class OutputLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.flush_time = datetime.datetime.now()

    def __enter__(self):
        sys.stdout, sys.stderr = self, self

    def __exit__(self, type, value, traceback):
        sys.stdout, sys.stderr = self.stdout, self.stderr

    def write(self, *args, **kwargs):
        self.log_file.write(*args, **kwargs)
        self.log_file.flush()
        return self.stdout.write(*args, **kwargs)

    def __getattr__(self, name):
        def inner(*args, **kwargs):
            getattr(self.log_file, name)(*args, **kwargs)
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
    start_time = datetime.datetime.now()
    def cleanup():
        if (datetime.datetime.now() - start_time).total_seconds() < 30:
            print('** NOT SAVING **')
            os.unlink(code_filename)
    try:
        with open(code_filename, 'x') as code_file:
            code_file.write(f'"""\n{description}\n"""\n\n{inspect.getsource(fn)}\n\n"""\n')
            code_file.flush()
            with OutputLogger(code_file):
                outputs = fn()
            code_file.write('"""\n')
    except Exception as e:
        cleanup()
        raise e
    cleanup()
    with open(f'log.txt', 'a+') as summary:
        summary.write(f'\n===\n{code_filename}\n\n{description}\n')
    return outputs


CONST_LIST = {}


def run_const(fn):
    code = inspect.getsource(fn)
    if code not in CONST_LIST:
        name = fn.__name__ if fn.__name__ != '<lambda>' else code
        print(f'eval: {name}')
        CONST_LIST[code] = fn()
    return CONST_LIST[code]


def optimize(model, opt_state, update, eval=lambda *k: '', iter_count=2000, seed=0):
    t = datetime.datetime.now()
    displayed = 0
    last = 0
    key_zero = jax.random.PRNGKey(0)
    interrupted = False
    try:
        for b, key in zipkey(range(iter_count), jax.random.PRNGKey(seed)):
            loss, model, opt_state = update(key, model, opt_state)
            spent = (datetime.datetime.now() - t).total_seconds()
            if b < 1 or spent > 5.0:
                speed = (b - last) / spent
                print(f'{displayed:02d} {b:04d} {float(loss):.3e} {speed:.1f}it/s ', end='')
                print(eval(model, key_zero))
                t = datetime.datetime.now()
                last = b
                displayed += 1
    except KeyboardInterrupt:
        print('interrupt')
        interrupted = True

    loss, unused_model, unused_opt_state = update(key_zero, model, opt_state)
    print(f'xx {b:04d} {float(loss):.3e} {eval(model, key_zero)} (done)')

    return model, opt_state, interrupted


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


class StopExecution(Exception):
    def _render_traceback_(self): return ['terminated']


def print_dtypes(vars):
    print('\n'.join(f'{k}: {v.dtype}{v.shape}' for k, v in vars.items() if hasattr(v, 'shape')))
    raise StopExecution()
