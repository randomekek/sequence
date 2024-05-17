"""
train TDLN missing digit 4, to see what happens

00 0000 0.29 13.3%
01 0007 0.20 6.3%
02 0025 0.12 40.7%
03 0044 0.07 71.2%
04 0063 0.07 74.2%
05 0082 0.06 75.0%
06 0100 0.06 76.2%
07 0119 0.06 76.7%
08 0138 0.05 77.7%
09 0157 0.05 78.0%
10 0175 0.05 79.2%
11 0194 0.06 78.7%
12 0212 0.05 79.7%
13 0231 0.05 79.7%
14 0249 0.05 80.6%
15 0267 0.05 80.5%
16 0285 0.05 81.2%
17 0303 0.04 81.5%
18 0321 0.04 81.4%
19 0339 0.05 81.9%
20 0357 0.04 81.6%
21 0376 0.04 82.2%
22 0394 0.04 82.5%
xx 0397 train 82.4% test 81.9% (done)

"""

def code():
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
      self.size = 1 - exp_rand(size_key, [D], minval=0.0001, maxval=0.001)
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
    D: int = 500
    LayerMid: int = 3
    Out: int = 28 * 28

    def __init__(self, key):
      keys = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = TDLN(keys[0], self.D, 0.0)
      self.layers = [(TDLN(k, self.D, 0.0), eqx.nn.LayerNorm([self.D])) for k in keys[1]]
      self.final = eqx.nn.Linear(self.D, self.Out, key=keys[2])

    def __call__(self, x):  # [L]
      x = einops.repeat(x, 'L -> L D', D=self.D)  # [L, D]
      x = self.dln(x)  # [L, D]
      for (layer, norm) in self.layers:
        x = norm(jax.nn.relu(layer(x) + x))
      x_last = jnp.real(x[-1, :])  # [D]
      return jax.nn.sigmoid(self.final(x_last))  # [D]

  train_images_without_4 = train_images[train_labels != 4]
  def get_batch(i):
    x = jax.random.choice(rnd[i], train_images_without_4, (40,))  # memory constraint
    return (x, x)

  def get_accuracy(x, y, cutoff):
    @eqx.filter_jit
    def fn(model):
      prediction = jax.vmap(model)(x[0:cutoff])
      return jnp.mean(jnp.abs(prediction - y[0:cutoff]) < 0.2)
    return fn

  @eqx.filter_value_and_grad
  def get_loss(model_dyn, model_const, x, y):
    model = eqx.combine(model_dyn, model_const)
    prediction = jax.vmap(model)(x)
    return jnp.mean((prediction - y) ** 2)

  @eqx.filter_jit
  def update(model, x, y, opt_state):
    model_dyn, model_const = eqx.partition(model, partition)
    loss, grads = get_loss(model_dyn, model_const, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

  get_train = get_accuracy(train_images, train_images, 50)
  get_test = get_accuracy(test_img, test_img, 400)
  model = Model(jax.random.PRNGKey(42))
  optimizer = optax.adam(1e-4)  # needs to be lower for more complex models
  partition = jax.tree_util.tree_map(lambda _: True, model)
  partition = eqx.tree_at(lambda t: t.dln, partition, replace=False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
