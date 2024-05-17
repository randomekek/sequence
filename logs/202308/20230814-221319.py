"""
try DLN using character level prediction

00 0000 0.26 0.0%
01 0011 0.24 0.0%
02 0029 0.24 0.0%
03 0047 0.24 0.0%
04 0065 0.24 0.0%
05 0083 0.23 0.0%
06 0101 0.22 0.4%
07 0119 0.22 1.0%
08 0137 0.21 0.7%
09 0155 0.21 1.6%
10 0173 0.20 5.5%
11 0191 0.17 14.3%
12 0209 0.16 20.9%
13 0227 0.15 27.1%
14 0245 0.14 31.9%
15 0263 0.12 38.3%
16 0281 0.11 43.0%
17 0299 0.11 45.2%
18 0317 0.11 46.5%
19 0335 0.10 48.3%
20 0353 0.09 51.5%
21 0371 0.08 55.8%
22 0389 0.14 35.8%
23 0408 0.12 45.8%
24 0426 0.10 50.1%
25 0444 0.08 54.0%
26 0462 0.07 58.0%
27 0481 0.06 62.4%
28 0499 0.05 67.0%
29 0517 0.04 71.3%
30 0535 0.03 75.3%
31 0553 0.03 79.3%
32 0571 0.02 83.9%
33 0589 0.02 86.7%
34 0607 0.01 91.2%
35 0625 0.01 94.1%
36 0643 0.01 95.9%
37 0661 0.01 97.2%
38 0680 0.01 98.3%
39 0699 0.01 98.7%
40 0718 0.01 99.1%
41 0737 0.01 99.2%
42 0756 0.01 99.3%
43 0775 0.01 99.4%
44 0794 0.00 99.5%
45 0813 0.00 99.6%
46 0832 0.00 99.6%
47 0851 0.00 99.6%
xx 0871 train 99.6% test 99.6% (done)

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
      self.size = 1 - exp_rand(size_key, [D], minval=0.001, maxval=0.1)
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
    D: int = 1*400
    Out: int = 1
    LayerMid: int = 1

    def __init__(self, key):
      D = self.D
      assert(self.D % self.Out == 0)
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

  # wikitext2_train = wikitext2('wiki.jsonl', 100*4000, 4000, 1)[0:1]
  # wikitext2_test = wikitext2_train
  def get_batch(i):
    x = jax.random.choice(rnd[i], wikitext2_train, (20,))  # memory constraint
    return (x, x)

  def get_accuracy(x, y, cutoff):
    @eqx.filter_jit
    def fn(model):
      prediction = jax.vmap(model)(x[0:cutoff])
      return jnp.mean(jnp.abs(prediction[:,:-1] - y[0:cutoff,1:]) < 0.2)
    return fn

  @eqx.filter_value_and_grad
  def get_loss(model_dyn, model_const, x, y):  # model, model, [batch, L, D]
    model = eqx.combine(model_dyn, model_const)
    prediction = jax.vmap(model)(x)
    return jnp.mean((prediction[:,:-1] - y[:,1:]) ** 2)

  @eqx.filter_jit
  def update(model, x, y, opt_state):
    model_dyn, model_const = eqx.partition(model, partition)
    loss, grads = get_loss(model_dyn, model_const, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

  get_train = get_accuracy(wikitext2_train, wikitext2_train, 10)
  get_test = get_accuracy(wikitext2_test, wikitext2_test, 50)
  model = Model(jax.random.PRNGKey(42))
  optimizer = optax.adam(1e-4)  # needs to be lower for more complex models
  partition = jax.tree_util.tree_map(lambda _: True, model)
  fixed = lambda dln: [dln.size, dln.theta]
  dlns = lambda t: sum([fixed(t.dln)] + [fixed(l[0]) for l in t.layers], start=[])
  partition = eqx.tree_at(dlns, partition, replace_fn=lambda n: False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
