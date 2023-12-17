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

def optimize(model, optimizer, get_batch, update, get_train, get_test, opt_state=None):
  if opt_state is None:
    opt_state = optimizer.init(model)
  logs = []

  def log(text, end='\n'):
    print(text, end=end)
    logs.append(text)
    logs.append(end)

  print('starting')
  t = datetime.datetime.now()
  displayed = 0
  try:
    for i in range(10000):
      x, y = get_batch(i)
      loss, model, opt_state = update(model, x, y, opt_state)
      if i < 1 or (datetime.datetime.now() - t).total_seconds() > 5.0:
        log(f'{displayed:02d} {i:04d} {loss:.3e} ', end='')
        log(f'{get_train(model)*100:0.1f}%')
        t = datetime.datetime.now()
        displayed += 1
  except KeyboardInterrupt:
    print('interrupt')

  log(f'xx {i:04d} train {get_train(model)*100:0.1f}% test {get_test(model)*100:0.1f}% (done)')

  return {
    'model': model,
    'opt_state': opt_state,
    'log': ''.join(logs),
  }
