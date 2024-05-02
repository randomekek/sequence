import datetime
import equinox as eqx
import inspect
import jax

DIR = 'logs/'


def run(fn, description):
    filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output = fn()
    filename = f'{DIR}{filename}.py'
    log = ''
    if output and 'log' in output:
        log = f'\n\n{output["log"]}'
    with open(filename, 'x') as code:
        code.write(f'"""\n{description}{log}\n"""\n\n{inspect.getsource(fn)}')
    with open(f'log.txt', 'a+') as summary:
        summary.write(f'===\n{filename}\n\n{description}\n\n')
    return output


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
        print(f'eval: {fn.__name__}')
        CONST_LIST[code] = fn()
    return CONST_LIST[code]


def optimize(model, optimizer, get_batch, partition, loss_fn, accuracy_fn, opt_state=None):
    if opt_state is None:
        opt_state = optimizer.init(model)
    logs = []

    def log(text, end='\n'):
        print(text, end=end)
        logs.append(text)
        logs.append(end)

    @eqx.filter_jit
    def update(model, x, y, opt_state):
        model_dyn, model_const = eqx.partition(model, partition)
        loss, grads = loss_fn(model_dyn, model_const, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print('starting')
    t = datetime.datetime.now()
    displayed = 0
    try:
        for i in range(10000):
            x, y = get_batch(i + 1)
            loss, model, opt_state = update(model, x, y, opt_state)
            if i < 1 or (datetime.datetime.now() - t).total_seconds() > 5.0:
                log(f'{displayed:02d} {i:04d} {loss:.3e} ', end='')
                log(f'{accuracy_fn(model)*100:0.1f}%')
                t = datetime.datetime.now()
                displayed += 1
    except KeyboardInterrupt:
        print('interrupt')

    log(f'xx {i:04d} accuracy {accuracy_fn(model)*100:0.1f}% (done)')

    return {
        'model': model,
        'opt_state': opt_state,
        'log': ''.join(logs),
    }
