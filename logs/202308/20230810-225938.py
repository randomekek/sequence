"""
train a faster TDLN with 1 level, without digit 4

00 0000 0.28 10.7%
01 0025 0.14 28.5%
02 0063 0.07 74.1%
03 0101 0.06 75.4%
04 0139 0.06 77.0%
05 0177 0.06 78.4%
06 0215 0.05 78.6%
07 0253 0.05 79.5%
08 0291 0.05 80.5%
09 0329 0.05 80.9%
10 0367 0.05 81.0%
11 0405 0.05 81.6%
12 0443 0.04 82.1%
13 0481 0.04 82.8%
14 0519 0.05 83.0%
15 0557 0.04 83.7%
16 0595 0.04 84.1%
17 0633 0.03 84.4%
18 0671 0.03 84.8%
19 0709 0.03 85.2%
20 0747 0.04 85.5%
21 0785 0.03 86.0%
22 0823 0.03 86.0%
23 0861 0.03 86.1%
24 0899 0.03 86.5%
25 0937 0.03 86.8%
26 0975 0.03 87.0%
27 1013 0.03 87.1%
28 1051 0.03 87.2%
29 1089 0.03 87.6%
30 1127 0.03 87.8%
31 1165 0.03 87.9%
32 1203 0.03 88.0%
33 1241 0.02 88.2%
34 1279 0.03 88.4%
35 1317 0.03 88.5%
36 1355 0.03 88.9%
37 1393 0.02 88.8%
38 1432 0.02 89.0%
39 1471 0.02 89.2%
40 1510 0.02 89.3%
41 1549 0.02 89.4%
42 1589 0.02 89.6%
43 1628 0.02 89.6%
xx 1650 train 89.6% test 89.1% (done)

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
    LayerMid: int = 1
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
