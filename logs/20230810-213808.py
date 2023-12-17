"""
unify DLN and TDLN, run TDLN, T=1

00 0000 0.24 4.9%
01 0052 0.09 51.3%
02 0110 0.06 73.0%
03 0169 0.06 73.6%
04 0228 0.06 75.1%
05 0286 0.05 75.9%
06 0344 0.05 76.9%
07 0403 0.04 78.3%
08 0461 0.04 79.4%
09 0519 0.04 81.0%
10 0577 0.03 82.1%
11 0635 0.04 82.6%
12 0693 0.03 83.8%
13 0752 0.03 84.6%
14 0810 0.03 85.2%
15 0868 0.03 85.9%
16 0926 0.03 86.5%
17 0984 0.03 87.2%
18 1043 0.03 87.7%
19 1101 0.02 88.2%
20 1160 0.02 88.7%
21 1218 0.02 89.1%
22 1277 0.02 89.6%
23 1335 0.02 89.9%
24 1393 0.02 90.3%
25 1451 0.02 90.6%
26 1509 0.02 90.8%
27 1567 0.01 91.2%
28 1625 0.02 91.4%
29 1683 0.02 91.6%
30 1741 0.02 91.8%
31 1799 0.01 91.9%
32 1858 0.01 92.2%
33 1916 0.01 92.4%
34 1975 0.01 92.6%
35 2033 0.01 92.8%
36 2092 0.01 93.0%
37 2151 0.01 93.0%
38 2210 0.01 93.2%
39 2269 0.01 93.4%
40 2329 0.01 93.6%
41 2389 0.01 93.7%
42 2449 0.01 93.9%
43 2509 0.01 94.0%
44 2569 0.01 94.2%
45 2629 0.01 94.2%
46 2689 0.01 94.4%
47 2748 0.01 94.5%
48 2808 0.01 94.6%
49 2868 0.01 94.7%
50 2928 0.01 94.8%
51 2988 0.01 94.9%
52 3048 0.01 95.0%
53 3108 0.01 95.0%
54 3167 0.01 95.1%
xx 3169 train 95.1% test 94.9% (done)

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
