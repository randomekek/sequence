"""
unify DLN and TDLN, run TDLN, T=1

00 0000 0.24 4.9%
01 0052 0.09 51.3%
02 0110 0.06 73.0%
03 0168 0.06 73.6%
04 0226 0.05 75.1%
05 0284 0.05 75.8%
06 0343 0.05 77.0%
07 0402 0.04 78.3%
08 0459 0.04 79.4%
09 0517 0.04 80.9%
10 0575 0.04 82.0%
11 0633 0.03 82.7%
12 0692 0.03 83.7%
13 0750 0.03 84.6%
14 0808 0.03 85.3%
15 0866 0.02 85.9%
16 0924 0.02 86.6%
17 0982 0.03 87.2%
18 1040 0.02 87.8%
19 1098 0.02 88.2%
20 1157 0.02 88.7%
xx 1192 train 88.9% test 88.6% (done)

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
      self.dln = TDLN(keys[0], self.D, 1000.0)
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
