"""
train a TDLN on wikitext2

00 0000 0.27 0.7%
01 0013 0.20 7.4%
02 0034 0.18 20.3%
03 0055 0.17 25.3%
04 0076 0.17 25.2%
05 0098 0.17 24.9%
06 0119 0.17 25.2%
07 0141 0.17 24.9%
08 0162 0.17 24.6%
09 0183 0.17 24.9%
10 0204 0.17 24.8%
11 0226 0.17 24.8%
12 0247 0.17 24.6%
13 0269 0.17 24.6%
14 0290 0.17 24.8%
15 0312 0.17 24.7%
16 0333 0.17 24.6%
17 0355 0.17 24.6%
18 0376 0.17 24.6%
19 0398 0.17 24.6%
20 0419 0.17 24.8%
21 0440 0.17 24.6%
22 0461 0.17 24.6%
23 0483 0.17 24.5%
24 0504 0.17 24.5%
25 0526 0.17 24.7%
26 0547 0.17 24.5%
27 0569 0.17 24.6%
28 0590 0.17 24.6%
29 0612 0.17 24.5%
30 0633 0.17 24.5%
31 0654 0.17 24.5%
32 0675 0.17 24.5%
33 0696 0.17 24.5%
34 0717 0.17 24.5%
35 0738 0.17 24.5%
36 0759 0.17 24.5%
37 0781 0.17 24.5%
38 0803 0.17 24.5%
39 0825 0.17 24.6%
40 0848 0.17 24.5%
41 0871 0.17 24.5%
42 0894 0.17 24.5%
43 0917 0.17 24.5%
44 0940 0.17 24.5%
45 0963 0.17 24.6%
46 0986 0.17 24.5%
47 1009 0.17 24.5%
48 1032 0.17 24.5%
49 1055 0.17 24.5%
50 1078 0.17 24.5%
51 1101 0.17 24.5%
52 1124 0.17 24.5%
53 1147 0.17 24.5%
54 1170 0.17 24.5%
55 1193 0.17 24.5%
56 1216 0.17 24.5%
57 1239 0.17 24.5%
58 1262 0.17 24.5%
59 1286 0.17 24.5%
60 1310 0.17 24.5%
61 1334 0.17 24.5%
62 1358 0.17 24.5%
63 1382 0.17 24.5%
64 1406 0.17 24.5%
65 1430 0.17 24.5%
66 1454 0.17 24.5%
67 1478 0.17 24.5%
68 1501 0.17 24.5%
69 1523 0.17 24.5%
xx 1540 train 24.5% test 24.4% (done)

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
    D: int = 4*8*20
    Out: int = 4*8
    LayerMid: int = 1

    def __init__(self, key):
      D = self.D
      assert(self.D % 32 == 0)
      dln_key, layer_keys, final_key = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = TDLN(dln_key, D, 0.0)
      self.layers = [(TDLN(k, D, 0.0), eqx.nn.LayerNorm([D])) for k in layer_keys]
      self.final = eqx.nn.Linear(D, self.Out, key=final_key)

    def __call__(self, x):  # [L, Out]
      x = einops.repeat(x, 'L d -> L (repeat d)', repeat=self.D//32)  # [L, D]
      x = self.dln(x)  # [L, D]
      for (dln, norm) in self.layers:
        x = norm(jax.nn.relu(dln(x) + x))  # [L, D]
      return jax.nn.sigmoid(jax.vmap(self.final)(x))  # [L, Out]

  # wikitext2_train = wikitext2('wiki.jsonl', 1000*2000*4*8, 2000, 4*8)
  # wikitext2_test = wikitext2('wiki-test.jsonl', 10*2000*4*8, 2000, 4*8)
  def get_batch(i):
    x = jax.random.choice(rnd[i], wikitext2_train, (20,))  # memory constraint
    return (x, x)

  def get_accuracy(x, y, cutoff):
    @eqx.filter_jit
    def fn(model):
      prediction = jax.vmap(model)(x[0:cutoff])
      return jnp.mean(jnp.abs(prediction[:-1] - y[0:cutoff][1:]) < 0.2)
    return fn

  @eqx.filter_value_and_grad
  def get_loss(model_dyn, model_const, x, y):  # model, model, [batch, L, D]
    model = eqx.combine(model_dyn, model_const)
    prediction = jax.vmap(model)(x)
    return jnp.mean((prediction[:][:-1] - y[:][1:]) ** 2)

  @eqx.filter_jit
  def update(model, x, y, opt_state):
    model_dyn, model_const = eqx.partition(model, partition)
    loss, grads = get_loss(model_dyn, model_const, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

  get_train = get_accuracy(wikitext2_train, wikitext2_train, 20)
  get_test = get_accuracy(wikitext2_test, wikitext2_test, 50)
  model = Model(jax.random.PRNGKey(42))
  optimizer = optax.adam(1e-4)  # needs to be lower for more complex models
  base_partition = jax.tree_util.tree_map(lambda _: True, model)
  fixed = lambda dln: [dln.size, dln.theta]
  dlns = lambda t: sum([fixed(t.dln)] + [fixed(l[0]) for l in t.layers], start=[])
  partition = eqx.tree_at(dlns, base_partition, replace_fn=lambda n: False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
