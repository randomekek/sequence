"""
try out tln

00 0000 0.25 4.9%
01 0061 0.08 62.2%
02 0129 0.06 73.3%
03 0199 0.05 75.2%
04 0268 0.05 76.9%
05 0338 0.05 78.4%
06 0407 0.04 79.9%
07 0476 0.04 81.2%
08 0545 0.03 82.0%
09 0614 0.03 82.6%
10 0683 0.03 83.8%
11 0752 0.03 84.5%
12 0822 0.03 85.4%
13 0892 0.03 85.8%
14 0961 0.02 86.6%
15 1030 0.02 87.2%
16 1100 0.02 87.8%
xx 1142 train 87.9% test 87.5% (done)

"""

def code():
  def exp_rand(key, shape, minval, maxval):
    return jnp.exp(jax.random.uniform(key, shape, minval=jnp.log(minval), maxval=jnp.log(maxval)))

  class DLN(eqx.Module):
    # L: input length
    # D: input dimension
    size: jax.Array # D
    theta: jax.Array # D
    D: int

    def __init__(self, key: jax.random.KeyArray, D: int):
      size_key, theta_key = jax.random.split(key, 2)
      self.size = 1 - exp_rand(size_key, [D], minval=0.0001, maxval=0.001)
      self.theta = exp_rand(theta_key, [D], 1/1000 * 2 * jnp.pi, 2 * jnp.pi)
      self.D = D

    def __call__(self, x):  # [L, D]
      x = jax.vmap(self.dimensionless, in_axes=(1, 0, 0), out_axes=1)(x, self.size, self.theta)
      return x

    def dimensionless(self, x, size, theta):  # [L]
      # do not differentiate size and theta jax.lax.stop_gradient(
      z = size * jnp.exp(1j * theta)
      def combine(a, b):
        pa, va = a
        pb, vb = b
        return jnp.stack([pa * pb, va * pb + vb])
      x = jnp.stack([jnp.ones(x.shape) * z, x])  # [2, L]
      x = jnp.take(jax.lax.associative_scan(combine, x, axis=1), 1, axis=0)  # [L]
      return x

  class TDLN(eqx.Module):
    # L: input length
    # D: input dimension
    size: jax.Array # D
    theta: jax.Array # D
    time: jax.Array # [D, D]
    D: int

    def __init__(self, key: jax.random.KeyArray, D: int):
      size_key, theta_key, time_key = jax.random.split(key, 3)
      self.size = 1 - exp_rand(size_key, [D], minval=0.0001, maxval=0.001)
      self.theta = exp_rand(theta_key, [D], 1/1000 * 2 * jnp.pi, 2 * jnp.pi)
      self.time = jax.random.normal(time_key, [D, D])
      self.D = D

    def __call__(self, x):  # [L, D]
      t = jax.nn.sigmoid(100 + einops.einsum(self.time, x, 'D b, L D -> L b'))  # [L, D]
      x = jax.vmap(self.dimensionless, in_axes=(1, 0, 0, 1), out_axes=1)(x, self.size, self.theta, t)
      return x

    def dimensionless(self, x, size, theta, t):  # [L], 1, 1, [L]
      z = jnp.power(size, t) * jnp.exp(1j * theta * t) # L
      x = jnp.append(jnp.array([1], dtype=x.dtype), t[0:-1]) * x
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
    LayerMid: int = 1
    Out: int = 28 * 28

    def __init__(self, key):
      keys = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = TDLN(keys[0], self.D)
      self.layers = [(eqx.nn.Linear(self.D, self.D, key=k), eqx.nn.LayerNorm([self.D])) for k in keys[1]]
      self.final = eqx.nn.Linear(self.D, self.Out, key=keys[2])

    def __call__(self, x):  # [L]
      x = einops.repeat(x, 'L -> L D', D=self.D)  # [L, D]
      x = self.dln(x)  # [L, D]
      x_last = jnp.real(x[-1, :])  # [D]
      for (layer, norm) in self.layers:
        x_last = norm(jax.nn.relu(layer(x_last) + x_last))
      return jax.nn.sigmoid(self.final(x_last))  # [D]

  def get_batch(i):
    x = jax.random.choice(rnd[i], train_images, (40,))  # memory constraint
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
