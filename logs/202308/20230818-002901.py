"""
try DLN using character level 256 way softmax

00 0000 0.02 0.0%
01 0007 0.02 0.0%
02 0018 0.01 0.0%
03 0029 0.01 0.0%
04 0040 0.01 0.0%
05 0051 0.01 0.0%
06 0062 0.01 0.0%
07 0073 0.01 0.0%
08 0084 0.01 0.0%
09 0095 0.01 0.0%
10 0106 0.01 0.0%
11 0117 0.01 0.0%
12 0128 0.01 0.0%
13 0139 0.01 0.0%
14 0150 0.01 0.0%
15 0161 0.01 0.0%
16 0172 0.01 0.0%
17 0183 0.01 0.0%
18 0194 0.01 0.0%
19 0205 0.01 0.0%
20 0216 0.01 0.0%
21 0227 0.01 0.0%
22 0238 0.01 0.0%
23 0249 0.01 0.0%
24 0260 0.01 0.0%
25 0271 0.01 0.0%
26 0282 0.01 0.0%
27 0293 0.01 0.0%
28 0304 0.01 0.0%
29 0315 0.01 0.0%
30 0326 0.01 0.0%
31 0337 0.01 0.0%
32 0348 0.01 0.0%
33 0359 0.01 0.0%
34 0370 0.01 0.0%
35 0381 0.01 0.0%
36 0392 0.01 0.0%
37 0403 0.01 0.0%
38 0414 0.01 0.0%
39 0425 0.01 0.0%
40 0436 0.01 0.0%
41 0447 0.01 0.0%
42 0458 0.01 0.0%
43 0469 0.01 0.0%
44 0480 0.01 0.0%
45 0491 0.01 0.0%
46 0502 0.01 0.0%
47 0513 0.01 0.0%
48 0524 0.01 0.0%
49 0535 0.01 0.0%
50 0546 0.01 0.0%
xx 0556 train 0.0% test 0.0% (done)

"""

def code():
  global model
  global wikitext2_train

  def wikitext2(file, byte_limit):
    def stream(byte_limit):
      with open(file, 'r') as f:
        for line in f:
          tokens = json.loads(line)['tokens']
          text = ' '.join(tokens).replace('\n ', '\n').replace('@@END@@', '\n')
          bytes = text.encode('utf-8')
          byte_limit -= len(bytes)
          if byte_limit <= 0:
            return
          yield bytes
    all_bytes = b''.join(stream(byte_limit))
    bytes_array = jnp.array(list(all_bytes), dtype=jnp.uint8)
    bytes_hot = jax.nn.one_hot(bytes_array, 256)
    return bytes_hot

  wikitext2_train = wikitext2('wiki.jsonl', 500000)

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
    In: int = 256
    D: int = 256*4
    Out: int = 256
    LayerMid: int = 1

    def __init__(self, key):
      D = self.D
      assert(self.D % self.In == 0)
      dln_key, layer_keys, final_key = utils.split_shape(key, [0, self.LayerMid, 0])
      self.dln = TDLN(dln_key, D, 0.0)
      self.layers = [(TDLN(k, D, 0.0), eqx.nn.LayerNorm([D])) for k in layer_keys]
      self.final = eqx.nn.Linear(D, self.Out, key=final_key)
      eqx.nn.MLP

    def __call__(self, x):  # [L, In]
      x = einops.repeat(x, 'L d -> L (repeat d)', repeat=self.D//self.In)  # [L, D]
      x = self.dln(x)  # [L, D]
      for (dln, norm) in self.layers:
        x = norm(jax.nn.relu(dln(x) + x))  # [L, D]
      return jax.nn.softmax(jax.vmap(self.final)(x))  # [L, Out]

  def get_batch(i):
    idx = jax.random.randint(rnd[i], shape=(100,), minval=0, maxval=wikitext2_train.shape[0]-1000)
    x = jnp.array([wikitext2_train[i:i+300] for i in idx])
    return (x, x)

  def get_accuracy(x):
    @eqx.filter_jit
    def fn(model):
      prediction = model(x)[:-1]
      truth = x[1:]
      return jnp.sum(jnp.equal(jnp.argmax(prediction), jnp.argmax(truth)))
    return fn

  @eqx.filter_value_and_grad
  def get_loss(model_dyn, model_const, x, y):  # model, model, [batch, L, D]
    model = eqx.combine(model_dyn, model_const)
    prediction = jax.vmap(model)(x)
    epsilon = 1e-10
    prediction = jnp.clip(prediction, epsilon, 1.0 - epsilon)[:,:-1]
    truth = y[:,1:]
    loss = -truth * jnp.log(prediction)
    return jnp.mean(loss)

  @eqx.filter_jit
  def update(model, x, y, opt_state):
    model_dyn, model_const = eqx.partition(model, partition)
    loss, grads = get_loss(model_dyn, model_const, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

  get_train = get_accuracy(wikitext2_train[1000:1500])
  get_test = get_train
  model = Model(jax.random.PRNGKey(42))
  optimizer = optax.adam(1e-4)  # needs to be lower for more complex models
  partition = jax.tree_util.tree_map(lambda _: True, model)
  fixed = lambda dln: [dln.size, dln.theta]
  dlns = lambda t: sum([fixed(t.dln)] + [fixed(l[0]) for l in t.layers], start=[])
  partition = eqx.tree_at(dlns, partition, replace_fn=lambda n: False)

  return utils.optimize(model, optimizer, get_batch, update, get_train, get_test)
