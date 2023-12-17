# %%
# imports
import jax
import jax.numpy as jnp
import equinox as eqx
import mnist
import optax
import datetime
import einops

jax.config.update("jax_debug_nans", True)
jnp.set_printoptions(suppress=True, precision=2, floatmode='fixed')
assert jax.devices()[0].device_kind == 'NVIDIA GeForce RTX 3060'

# %%
# data source
order = jnp.array(range(784))
def shuffle(images):
  # shuffle works beter
  return jax.random.shuffle(jax.random.PRNGKey(42), images, axis=1)
  return images

train_images = shuffle(mnist.train_images().reshape((-1, 784))[:, order] / 255)  # [B, 784]
train_labels = mnist.train_labels()
train_labels_hot = jax.nn.one_hot(train_labels, 10)

test_img = shuffle(mnist.test_images().reshape((-1, 784))[:, order] / 255)
test_lbl = mnist.test_labels()

# %%
# DLN

class DLN(eqx.Module):
  # L: input length
  # D: input dimension
  size: jax.Array # D
  theta: jax.Array # D
  D: int

  def __init__(self, key: jax.random.KeyArray, D: int):
    size_key, theta_key = jax.random.split(key, 2)
    self.size = jax.random.normal(size_key, [D])
    self.theta = jax.random.normal(theta_key, [D]) * (2*jnp.pi) / 500.0
    self.D = D

  def __call__(self, x):  # [L, D]
    x = jax.vmap(self.dimensionless, in_axes=(1, 0, 0), out_axes=1)(x, self.size, self.theta)
    return x

  def dimensionless(self, x, size, theta):  # [L],
    # TODO: theta adjustment for size ~ 0
    z = jnp.exp(-jnp.exp(size) + 1j * theta)
    def combine(a, b):
      pa, va = a
      pb, vb = b
      return jnp.stack([pa * pb, va * pb + vb])
    x = jnp.stack([jnp.ones(x.shape) * z, x])  # [2, L]
    x = jnp.take(jax.lax.associative_scan(combine, x, axis=1), 1, axis=0)  # [L]
    return x

@eqx.filter_value_and_grad
def calculate_loss(model, x, y):
  prediction = jax.vmap(model)(x)
  return -jnp.mean(jnp.log(0.001+jnp.sum(prediction * y)))

@eqx.filter_jit
def update(model, x, y, opt_state):
  loss, grads = calculate_loss(model, x, y)
  updates, opt_state = optimizer.update(grads, opt_state)
  model = eqx.apply_updates(model, updates)
  return loss, model, opt_state

@eqx.filter_jit
def accuracy(model, x, y):
  prediction = jax.vmap(model)(x)
  return jnp.mean(jnp.argmax(prediction, axis=1) == y)

class Model(eqx.Module):
  dlns: list[(eqx.Module, eqx.Module)]
  final: eqx.nn.Linear
  Layers: int = 1
  D: int = 1000

  def __init__(self, key):
    final_key, *layer_keys = jax.random.split(key, self.Layers+1)
    self.dlns = [(DLN(k, self.D), eqx.nn.Linear(self.D, self.D, key=k)) for k in layer_keys]
    self.final = eqx.nn.Linear(self.D, 10, key=final_key)

  def __call__(self, x):  # [L]
    x = einops.repeat(x, 'L -> L D', D=self.D)  # [L, D]
    for (dln, linear) in self.dlns:
      x = jax.vmap(linear)(dln(x)) + x
      x = jax.nn.relu(jnp.real(x))
    x = self.final(jnp.real(x[-1, :]))  # [D]
    x = jax.nn.softmax(x)
    return x

model = Model(jax.random.PRNGKey(42))
optimizer = optax.adam(1e-5)  # needs to be lower for more complex models
opt_state = optimizer.init(model)
t = datetime.datetime.now()
steps = 100000
batch_size = (40,)  # memory constraint
rnd = jax.random.split(jax.random.PRNGKey(0), steps)

print('starting')
try:
  for i in range(0, steps):
    x = jax.random.choice(rnd[i], train_images, batch_size)
    y = jax.random.choice(rnd[i], train_labels_hot, batch_size)
    loss, model, opt_state = update(model, x, y, opt_state)
    if i < 2 or (datetime.datetime.now() - t).total_seconds() > 5.0:
      print(f'{i:04d} {loss:0.2f} ', end='')
      train = accuracy(model, train_images[0:50,:], train_labels[0:50])
      print(f'{train*100:0.1f}%')
      t = datetime.datetime.now()
except KeyboardInterrupt:
    print(f'interrupt\n{i:04d} ', end='')

print_size = 200
train = accuracy(model, train_images[0:print_size], train_labels[0:print_size])
test = accuracy(model, test_img[0:print_size], test_lbl[0:print_size])
print(f'train {train*100:0.1f}% test {test*100:0.1f}% (done)')
