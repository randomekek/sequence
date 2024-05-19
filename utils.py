import datetime
import inspect
import jax
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


def run(fn, description):
    root = pathlib.Path('logs')
    now = lambda fmt: datetime.datetime.now().strftime(fmt)
    folder = root.joinpath(now('%Y%m'))
    folder.mkdir(exist_ok=True)
    code_filename = folder.joinpath(now('%Y%m%d-%H%M%S') + '.py')
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
    try:
        for b, key in zipkey(range(iter_count), jax.random.PRNGKey(seed)):
            loss, model, opt_state = update(key, model, opt_state)
            if b < 1 or (datetime.datetime.now() - t).total_seconds() > 5.0:
                print(f'{displayed:02d} {b:04d} {loss:.3e} ', end='')
                print(f'{accuracy_fn(model, key)*100:0.1f}%')
                t = datetime.datetime.now()
                displayed += 1
    except KeyboardInterrupt:
        print('interrupt')

    print(f'xx {b:04d} accuracy {accuracy_fn(model, jax.random.PRNGKey(0))*100:0.1f}% (done)')

    return model, opt_state


def param_count(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


def zipkey(items, key):
    return zip(items, jax.random.split(key, len(items)))
