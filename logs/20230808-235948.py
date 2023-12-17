"""
see if it is possible to train memory too

00 0000 0.24 5.0%
01 0064 0.19 6.0%
02 0144 0.14 13.0%
03 0225 0.11 24.5%
04 0306 0.09 38.0%
05 0387 0.08 52.3%
06 0468 0.08 63.1%
07 0549 0.07 68.7%
08 0630 0.08 70.9%
09 0711 0.07 71.4%
10 0791 0.08 71.6%
11 0872 0.07 71.8%
12 0948 0.07 71.8%
13 1025 0.07 72.1%
14 1105 0.07 72.0%
15 1186 0.07 72.0%
16 1266 0.06 72.4%
17 1345 0.07 72.5%
18 1425 0.06 72.5%
19 1506 0.07 72.6%
20 1587 0.06 72.9%
21 1668 0.07 73.3%
22 1748 0.06 73.4%
23 1829 0.06 73.6%
xx 1863 train 73.6% test 73.1% (done)
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
      # do not differentiate size and theta jax.lax.stop_gradient
      z = (size * jnp.exp(1j * theta))
      def combine(a, b):
        pa, va = a
        pb, vb = b
        return jnp.stack([pa * pb, va * pb + vb])
      x = jnp.stack([jnp.ones(x.shape) * z, x])  # [2, L]
      x = jnp.take(jax.lax.associative_scan(combine, x, axis=1), 1, axis=0)  # [L]
      return x

  class Model(eqx.Module):
    dln: eqx.Module
    layers: list[eqx.nn.Linear]
    final: eqx.nn.Linear
    D: int = 50
    LayerMid: int = 1
    Out: int = 28 * 28

    def __init__(self, key):
      keys = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = DLN(keys[0], self.D)
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

  @eqx.filter_value_and_grad
  def get_loss(model, x, y):
    prediction = jax.vmap(model)(x)
    return jnp.mean((prediction - y) ** 2)

  def get_accuracy(x, y, cutoff):
    @eqx.filter_jit
    def fn(model):
      prediction = jax.vmap(model)(x[0:cutoff])
      return jnp.mean(jnp.abs(prediction - y[0:cutoff]) < 0.2)
    return fn

  get_train = get_accuracy(train_images, train_images, 50)
  get_test = get_accuracy(test_img, test_img, 400)

  model = Model(jax.random.PRNGKey(42))
  optimizer = optax.adam(1e-4)  # needs to be lower for more complex models

  return utils.optimize(model, optimizer, get_batch, get_loss, get_train, get_test)
