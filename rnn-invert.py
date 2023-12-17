# 2023-08-06
# Make a random DLN and learn a network to invert it
# for random fixed DLN(x_t) -> train MLP(DLN(x)) ~ x

# %%
# imports
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import json
import matplotlib.pyplot as plt
import mnist
import optax
import utils

jax.config.update("jax_debug_nans", True)
jnp.set_printoptions(suppress=True, precision=2, floatmode='fixed')
assert jax.devices()[0].device_kind == 'NVIDIA GeForce RTX 3060'
rnd = jax.random.split(jax.random.PRNGKey(0), 100000)

# %%
# data source mnist
def shuffle(images):
  # return jax.random.shuffle(rnd[0], images, axis=1)
  return images

train_images = shuffle(mnist.train_images().reshape((-1, 784)) / 255)  # [B, 784]
train_labels = mnist.train_labels()
train_labels_hot = jax.nn.one_hot(train_labels, 10)

test_img = shuffle(mnist.test_images().reshape((-1, 784)) / 255)
test_lbl = mnist.test_labels()


# %%
# TDLN
# y' = exp(A, t) y + t B x
# t = C x

model = None
wikitext2_train = None

def code():
  global model
  global wikitext2_train

  def wikitext2(file, byte_limit):
    def stream(byte_limit):
      with open(file, 'r') as f:
        for line in f:
          tokens = json.loads(line)['tokens']
          text = ' '.join(tokens).replace('\n ', '\n').replace('@@END@@', '\n')
          bytes = text.encode('utf-8')
          byte_limit -= len(bytes)
          if byte_limit <= 0:
            return
          yield bytes
    all_bytes = b''.join(stream(byte_limit))
    bytes_array = jnp.array(list(all_bytes), dtype=jnp.uint8)
    bytes_hot = jax.nn.one_hot(bytes_array, 256)
    return bytes_hot

  wikitext2_train = wikitext2('wiki.jsonl', 500000)

  def exp_rand(key, shape, minval, maxval):
    return jnp.exp(jax.random.uniform(key, shape, minval=jnp.log(minval), maxval=jnp.log(maxval)))

  class TDLN(eqx.Module):
    # L: input length
    # D: input dimension
    size: jax.Array # D
    theta: jax.Array # D
    time: eqx.nn.Linear # [D, D]
    data: eqx.nn.Linear # [D, D]
    D: int
    time_bias: float

    def __init__(self, key: jax.random.KeyArray, D: int, time_bias: float):
      size_key, theta_key, time_key, data_key = jax.random.split(key, 4)
      self.size = 1 - exp_rand(size_key, [D], minval=0.001, maxval=0.1)
      self.theta = exp_rand(theta_key, [D], 1/1000 * 2 * jnp.pi, 2 * jnp.pi)
      self.time = eqx.nn.Linear(D, D, key=time_key)
      self.data = eqx.nn.Linear(D, D, key=data_key)
      self.D = D
      self.time_bias = time_bias

    def __call__(self, x):  # [L, D]
      t = jax.nn.sigmoid(self.time_bias + jax.vmap(self.time)(x))  # [L, D]
      x = jax.vmap(self.data)(x)  # [L, D]
      x = jax.vmap(self.dimensionless, in_axes=(1, 0, 0, 1), out_axes=1)(x, self.size, self.theta, t)
      return jnp.real(x)

    def dimensionless(self, x, size, theta, t):  # [L], 1, 1, [L]
      z = jnp.power(size, t) * jnp.exp(1j * theta * t) # L
      x = t * x
      def combine(a, b):
        pa, va = a
        pb, vb = b
        return jnp.stack([pa * pb, va * pb + vb])
      x = jnp.stack([z, x])  # [2, L]
      x = jnp.take(jax.lax.associative_scan(combine, x, axis=1), 1, axis=0)  # [L]
      return x

  class Model(eqx.Module):
    dln: eqx.Module
    layers: list[eqx.nn.Linear]
    final: eqx.nn.Linear
    In: int = 256
    D: int = 256*8
    Out: int = 256
    LayerMid: int = 1

    def __init__(self, key):
      D = self.D
      assert(self.D % self.In == 0)
      dln_key, layer_keys, final_key = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = TDLN(dln_key, D, 0.0)
      self.layers = [(TDLN(k, D, 0.0), eqx.nn.LayerNorm([D])) for k in layer_keys]
      self.final = eqx.nn.Linear(D, self.Out, key=final_key)
      eqx.nn.MLP

    def __call__(self, x):  # [L, In]
      x = einops.repeat(x, 'L d -> L (repeat d)', repeat=self.D//self.In)  # [L, D]
      x = self.dln(x)  # [L, D]
      for (dln, norm) in self.layers:
        x = jax.vmap(norm)(jax.nn.relu(dln(x) + x))  # [L, D]
      return jax.nn.softmax(jax.vmap(self.final)(x))  # [L, Out]

  def get_batch(i):
    idx = jax.random.randint(rnd[i], shape=(10,), minval=0, maxval=wikitext2_train.shape[0]-1000)
    x = jnp.array([wikitext2_train[i:i+300] for i in idx])
    return (x, x)

  def get_accuracy(x):
    @eqx.filter_jit
    def fn(model):
      prediction = model(x)[:-1]
      truth = x[1:]
      return jnp.sum(jnp.equal(jnp.argmax(prediction), jnp.argmax(truth)))
    return fn

  @eqx.filter_value_and_grad
  def get_loss(model_dyn, model_const, x, y):  # model, model, [batch, L, D]
    model = eqx.combine(model_dyn, model_const)
    prediction = jax.vmap(model)(x)
    epsilon = 1e-10
    prediction = jnp.clip(prediction, epsilon, 1.0 - epsilon)[:,:-1]
    truth = y[:,1:]
    loss = -truth * jnp.log(prediction)
    return jnp.mean(loss)

  @eqx.filter_jit
  def update(model, x, y, opt_state):
    model_dyn, model_const = eqx.partition(model, partition)
    loss, grads = get_loss(model_dyn, model_const, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

  get_train = get_accuracy(wikitext2_train[1000:1500])
  get_test = get_train
  model = Model(jax.random.PRNGKey(42))
  optimizer = optax.adam(1e-4)  # needs to be lower for more complex models
  partition = jax.tree_util.tree_map(lambda _: True, model)
  fixed = lambda dln: [dln.size, dln.theta]
  dlns = lambda t: sum([fixed(t.dln)] + [fixed(l[0]) for l in t.layers], start=[])
  partition = eqx.tree_at(dlns, partition, replace_fn=lambda n: False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)

output = utils.run(code, '''try DLN using character level 256 way softmax''')
model = output['model']

# %%
# show memorized images
plt.figure(figsize = (3, 30))
plt.imshow(jnp.hstack([test_img[0:50].reshape(-1,28), jax.vmap(model)(test_img[0:50]).reshape(-1, 28)]))

# %%
# interpolate across two latent spaces
a = test_img[9]
b = test_img[6]
def m(c):
  return b + (a-b)*c
mid = jax.vmap(m)(jnp.arange(start=-0.5, stop=1.5, step=0.1))
plt.figure(figsize = (3, 30))
plt.imshow(jnp.vstack(jax.vmap(model)(mid).reshape(-1, 28)))

# %%
# show predicted words
def chars(a):
  return bytes(jnp.argmax(a, axis=1).tolist()).decode('utf-8')

def enc(a):
  bytes_array = jnp.array(list(a.encode('utf-8')), dtype=jnp.uint8)
  return jax.nn.one_hot(bytes_array, 256)

line = enc('combining could be understood as the addition of')
print(chars(line[1:]) + '\n\n' + chars(model(line)) + '\n\n')

start = 75500
line = wikitext2_train[start:start+100]
print(chars(line[1:]) + '\n\n' + chars(model(line)))

# %%
