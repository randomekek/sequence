"""
train a TDLN on wikitext2, remove partition, reduce adam, fix accuracy fn

00 0000 0.26 1.3%
01 0008 0.23 2.3%
02 0023 0.21 6.0%
03 0038 0.20 9.1%
04 0054 0.19 12.4%
05 0070 0.19 14.9%
06 0086 0.19 16.8%
07 0102 0.18 18.3%
08 0117 0.18 19.3%
09 0132 0.18 20.2%
10 0147 0.18 20.8%
11 0162 0.18 21.4%
12 0177 0.18 22.0%
13 0192 0.18 22.6%
14 0207 0.18 22.7%
15 0222 0.18 23.0%
16 0237 0.18 23.5%
17 0252 0.18 23.9%
18 0267 0.17 24.3%
19 0282 0.18 24.6%
20 0297 0.17 25.3%
21 0312 0.17 25.1%
22 0327 0.18 25.6%
23 0342 0.18 25.6%
24 0357 0.17 25.6%
25 0372 0.17 25.6%
26 0387 0.17 25.5%
27 0402 0.17 25.9%
28 0417 0.17 25.4%
29 0432 0.17 26.0%
30 0447 0.17 26.1%
31 0462 0.17 26.3%
32 0477 0.17 26.4%
33 0492 0.17 26.4%
34 0507 0.17 26.3%
35 0522 0.17 26.6%
36 0537 0.17 26.7%
37 0552 0.17 26.4%
38 0567 0.17 26.1%
39 0582 0.17 26.6%
40 0597 0.17 26.5%
41 0613 0.17 26.4%
42 0629 0.17 26.1%
43 0645 0.17 26.3%
44 0661 0.17 26.2%
45 0677 0.17 26.8%
46 0693 0.17 26.7%
47 0709 0.17 26.4%
48 0725 0.17 26.6%
49 0741 0.17 26.1%
50 0757 0.17 26.0%
51 0773 0.17 26.2%
52 0789 0.17 26.0%
53 0805 0.17 26.2%
54 0821 0.17 26.1%
55 0837 0.17 25.9%
56 0853 0.17 26.1%
57 0869 0.17 26.3%
58 0885 0.17 26.1%
59 0901 0.17 26.1%
60 0917 0.17 25.9%
61 0933 0.17 25.9%
62 0949 0.17 26.3%
63 0965 0.17 26.0%
64 0981 0.17 26.1%
65 0997 0.17 25.8%
66 1013 0.17 25.9%
67 1029 0.17 25.8%
68 1045 0.17 25.7%
69 1061 0.17 25.8%
70 1077 0.17 25.6%
71 1093 0.17 25.9%
72 1109 0.17 26.0%
73 1125 0.17 25.8%
74 1141 0.17 25.7%
75 1157 0.17 25.5%
76 1173 0.17 25.6%
77 1189 0.17 25.6%
78 1205 0.17 25.6%
79 1221 0.17 25.8%
80 1237 0.17 25.7%
81 1253 0.17 25.7%
82 1269 0.17 25.7%
83 1285 0.17 25.6%
84 1300 0.17 25.8%
85 1316 0.17 25.7%
86 1332 0.17 25.5%
xx 1337 train 25.6% test 25.5% (done)

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
    D: int = 8*50
    Out: int = 8
    LayerMid: int = 1

    def __init__(self, key):
      D = self.D
      assert(self.D % 8 == 0)
      dln_key, layer_keys, final_key = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = TDLN(dln_key, D, 0.0)
      self.layers = [(TDLN(k, D, 0.0), eqx.nn.LayerNorm([D])) for k in layer_keys]
      self.final = eqx.nn.Linear(D, self.Out, key=final_key)

    def __call__(self, x):  # [L, Out]
      x = einops.repeat(x, 'L d -> L (repeat d)', repeat=self.D//self.Out)  # [L, D]
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
      return jnp.mean(jnp.abs(prediction[:][:-1] - y[0:cutoff][1:]) < 0.2)
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
  optimizer = optax.adam(1e-5)  # needs to be lower for more complex models
  partition = jax.tree_util.tree_map(lambda _: True, model)
  # fixed = lambda dln: [dln.size, dln.theta]
  # dlns = lambda t: sum([fixed(t.dln)] + [fixed(l[0]) for l in t.layers], start=[])
  # partition = eqx.tree_at(dlns, partition, replace_fn=lambda n: False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
