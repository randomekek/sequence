"""
run TDLN, without T bias, 3 layers of TDLN

00 0000 0.29 13.2%
01 0006 0.21 5.9%
02 0024 0.12 37.6%
03 0042 0.08 69.3%
04 0060 0.07 73.9%
05 0078 0.06 74.2%
06 0096 0.06 75.7%
07 0114 0.06 76.5%
08 0132 0.05 77.4%
09 0150 0.06 77.8%
10 0168 0.05 78.5%
11 0186 0.05 78.6%
12 0204 0.05 79.6%
13 0222 0.05 79.4%
14 0240 0.04 79.4%
15 0258 0.05 79.8%
16 0276 0.04 80.2%
17 0294 0.05 81.2%
18 0312 0.05 80.5%
19 0330 0.04 81.5%
20 0348 0.05 81.7%
21 0366 0.05 82.4%
22 0384 0.04 81.8%
23 0402 0.04 82.3%
24 0420 0.04 82.4%
25 0438 0.04 82.9%
26 0456 0.04 83.1%
27 0474 0.04 83.0%
28 0492 0.04 83.2%
29 0510 0.04 83.9%
30 0528 0.04 84.2%
31 0546 0.04 84.2%
32 0564 0.03 84.4%
33 0582 0.04 84.5%
34 0600 0.04 84.2%
35 0618 0.04 84.5%
36 0636 0.03 84.8%
37 0654 0.04 84.6%
38 0673 0.04 84.8%
39 0692 0.03 84.9%
40 0711 0.04 85.5%
41 0730 0.03 85.6%
42 0749 0.04 85.6%
43 0768 0.03 85.7%
44 0787 0.03 86.1%
45 0806 0.03 86.3%
46 0825 0.04 86.1%
47 0844 0.03 86.0%
48 0863 0.03 86.5%
49 0882 0.03 86.1%
50 0901 0.03 86.5%
51 0920 0.03 86.5%
52 0940 0.03 86.8%
53 0960 0.03 86.4%
54 0980 0.03 87.0%
55 1000 0.03 86.9%
56 1020 0.03 87.3%
57 1040 0.03 87.5%
58 1060 0.03 87.5%
59 1080 0.03 87.4%
60 1100 0.03 87.5%
61 1120 0.03 87.4%
62 1140 0.03 87.7%
63 1160 0.03 87.8%
64 1180 0.03 87.8%
65 1199 0.03 88.1%
66 1218 0.03 87.8%
67 1237 0.03 88.2%
68 1256 0.03 88.1%
69 1275 0.03 88.1%
70 1295 0.03 88.2%
71 1315 0.03 88.6%
72 1335 0.02 88.6%
73 1355 0.02 88.4%
xx 1363 train 88.5% test 87.8% (done)

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
