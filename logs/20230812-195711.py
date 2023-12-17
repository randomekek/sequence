"""
retrain TDLN with bug fix

00 0000 0.22 1.4%
01 0008 0.21 1.4%
02 0023 0.20 1.7%
03 0038 0.19 2.2%
04 0053 0.19 4.1%
05 0068 0.18 7.1%
06 0083 0.18 11.6%
07 0098 0.18 16.1%
08 0113 0.18 19.8%
09 0128 0.18 22.3%
10 0143 0.18 23.7%
11 0158 0.17 24.7%
12 0173 0.17 25.1%
13 0188 0.17 25.3%
14 0203 0.17 25.2%
15 0218 0.17 25.3%
16 0233 0.17 25.3%
17 0248 0.17 25.3%
18 0263 0.17 25.3%
19 0278 0.17 25.3%
20 0293 0.17 25.2%
21 0308 0.17 25.2%
22 0323 0.17 25.2%
23 0338 0.17 25.2%
24 0353 0.17 25.2%
25 0368 0.17 25.1%
26 0383 0.17 25.0%
27 0398 0.17 25.1%
28 0413 0.17 25.0%
29 0428 0.17 24.9%
30 0443 0.17 24.9%
31 0458 0.17 24.8%
32 0473 0.17 24.8%
33 0488 0.17 24.9%
34 0503 0.17 24.8%
35 0518 0.17 24.7%
36 0533 0.17 24.8%
37 0548 0.17 24.7%
38 0564 0.17 24.7%
39 0580 0.17 24.7%
40 0596 0.17 24.7%
41 0612 0.17 24.6%
42 0628 0.17 24.6%
43 0644 0.17 24.5%
44 0660 0.17 24.6%
45 0676 0.17 24.6%
46 0692 0.17 24.6%
xx 0706 train 24.6% test 24.7% (done)

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
  optimizer = optax.adam(1e-5)  # needs to be lower for more complex models
  partition = jax.tree_util.tree_map(lambda _: True, model)
  fixed = lambda dln: [dln.size, dln.theta]
  dlns = lambda t: sum([fixed(t.dln)] + [fixed(l[0]) for l in t.layers], start=[])
  partition = eqx.tree_at(dlns, partition, replace_fn=lambda n: False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
