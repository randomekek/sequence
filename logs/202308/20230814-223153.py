"""
try DLN using character level prediction, this time using binary cross entropy

00 0000 0.72 0.0%
01 0011 0.69 0.0%
02 0029 0.69 0.0%
03 0047 0.68 0.0%
04 0065 0.68 0.0%
05 0083 0.67 0.0%
06 0101 0.66 0.0%
07 0119 0.61 0.8%
08 0137 0.59 8.8%
09 0155 0.54 21.8%
10 0173 0.52 27.0%
11 0191 0.50 34.8%
12 0209 0.50 34.3%
13 0228 0.48 36.9%
14 0246 0.47 40.1%
15 0265 0.46 42.7%
16 0284 0.45 43.7%
17 0304 0.44 42.9%
18 0324 0.44 42.8%
19 0343 0.43 44.0%
20 0361 0.43 44.3%
21 0379 0.44 45.2%
22 0397 0.42 46.1%
23 0415 0.42 45.9%
24 0433 0.42 46.7%
25 0452 0.40 48.2%
26 0470 0.41 47.6%
27 0488 0.41 48.0%
28 0506 0.40 47.8%
29 0524 0.39 47.8%
30 0543 0.40 49.7%
31 0561 0.39 49.3%
32 0579 0.39 50.2%
33 0597 0.38 49.9%
34 0615 0.38 50.2%
35 0633 0.38 51.0%
36 0651 0.38 50.1%
37 0669 0.38 50.8%
38 0688 0.38 50.3%
39 0707 0.37 51.9%
40 0727 0.38 52.0%
41 0746 0.37 53.4%
42 0765 0.37 53.0%
43 0785 0.37 53.3%
44 0805 0.37 51.7%
45 0825 0.37 53.4%
46 0845 0.35 53.7%
47 0865 0.36 53.6%
xx 0873 train 54.1% test 53.9% (done)

"""

def code():
  global model

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
    epsilon = 1e-15
    prediction = jnp.clip(prediction, epsilon, 1.0 - epsilon)[:,:-1]
    truth = y[:,1:]
    loss = -(truth * jnp.log(prediction) + (1 - truth) * jnp.log(1 - prediction))
    return jnp.mean(loss)

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
