"""
verify increasing memory size actually memorizes

00 0000 0.25 5.1%
01 0064 0.08 66.6%
02 0136 0.06 73.4%
03 0207 0.05 75.0%
04 0278 0.06 76.5%
05 0349 0.04 78.2%
06 0421 0.04 79.6%
07 0493 0.04 81.2%
08 0565 0.04 81.9%
09 0636 0.03 82.7%
10 0709 0.03 83.7%
11 0781 0.03 84.6%
12 0853 0.03 85.4%
13 0924 0.02 86.0%
14 0996 0.03 86.6%
15 1068 0.02 87.3%
16 1139 0.02 87.8%
17 1211 0.02 88.3%
18 1283 0.02 88.8%
19 1354 0.02 89.2%
20 1425 0.02 89.6%
21 1497 0.02 89.9%
22 1569 0.02 90.4%
23 1641 0.02 90.7%
24 1713 0.02 90.9%
25 1785 0.01 91.3%
26 1857 0.01 91.4%
27 1929 0.01 91.7%
28 2000 0.02 91.8%
29 2072 0.01 92.2%
30 2144 0.01 92.2%
31 2216 0.01 92.6%
32 2288 0.01 92.6%
33 2360 0.01 92.8%
34 2432 0.01 92.9%
35 2504 0.01 93.1%
36 2576 0.01 93.2%
37 2648 0.01 93.5%
38 2721 0.01 93.6%
39 2794 0.01 93.6%
40 2867 0.01 93.7%
41 2940 0.01 93.8%
xx 3009 train 93.9% test 93.6% (done)

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

  class Model(eqx.Module):
    dln: eqx.Module
    layers: list[eqx.nn.Linear]
    final: eqx.nn.Linear
    D: int = 500
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
  partition = eqx.tree_at(lambda t: t.dln, jax.tree_util.tree_map(lambda _: True, model), replace=False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
