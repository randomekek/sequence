import datetime
import equinox as eqx
import inspect
import jax

DIR = 'logs/'


def run(fn, description):
    filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    outputs = fn()
    filename = f'{DIR}{filename}.py'
    log = ''
    for param in outputs:
        if type(param) == dict and 'log' in param:
            log = f'\n\n{param["log"]}'
    with open(filename, 'x') as code:
        code.write(f'"""\n{description}{log}\n"""\n\n{inspect.getsource(fn)}')
    with open(f'log.txt', 'a+') as summary:
        summary.write(f'===\n{filename}\n\n{description}\n\n')
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


def optimize(model, opt_state, update, accuracy_fn):
    logs = []

    def log(text, end='\n'):
        print(text, end=end)
        logs.append(text + end)

    print('starting')
    t = datetime.datetime.now()
    displayed = 0
    keys = jax.random.split(jax.random.PRNGKey(0), 10000)
    try:
        for b, key in zip(range(10000), keys):
            loss, model, opt_state = update(key, model, opt_state)
            if b < 1 or (datetime.datetime.now() - t).total_seconds() > 5.0:
                log(f'{displayed:02d} {b:04d} {loss:.3e} ', end='')
                log(f'{accuracy_fn(model, key)*100:0.1f}%')
                t = datetime.datetime.now()
                displayed += 1
    except KeyboardInterrupt:
        print('interrupt')

    log(f'xx {b:04d} accuracy {accuracy_fn(model, keys[0])*100:0.1f}% (done)')

    return model, opt_state, {'logs': logs}


def param_count(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))
