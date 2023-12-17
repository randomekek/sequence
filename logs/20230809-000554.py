"""
compare against similarly long run with fixed memory

00 0000 0.24 5.0%
01 0074 0.21 6.1%
02 0158 0.16 10.3%
03 0242 0.12 19.9%
04 0326 0.10 35.1%
05 0410 0.09 50.4%
06 0494 0.08 61.3%
07 0579 0.08 67.2%
08 0663 0.07 69.8%
09 0747 0.07 71.0%
10 0832 0.07 71.6%
11 0917 0.06 71.9%
12 1001 0.06 72.1%
13 1085 0.07 72.0%
14 1170 0.06 72.1%
15 1254 0.06 72.3%
16 1338 0.07 72.4%
17 1422 0.07 72.5%
18 1506 0.07 72.5%
19 1591 0.06 72.6%
20 1675 0.07 72.8%
21 1759 0.06 73.0%
22 1842 0.06 73.1%
23 1926 0.06 73.1%
xx 1966 train 73.3% test 72.6% (done)
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
      # do not differentiate size and theta
      z = jax.lax.stop_gradient(size * jnp.exp(1j * theta))
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
