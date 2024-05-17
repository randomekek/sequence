"""
try DLN using character level 256 way softmax

00 0000 2.171e-02 0.0%
01 0029 1.197e-02 0.0%
02 0069 1.193e-02 0.0%
03 0109 1.188e-02 0.0%
04 0148 1.214e-02 0.0%
05 0188 1.100e-02 0.0%
06 0228 9.870e-03 0.0%
07 0267 9.821e-03 0.0%
08 0306 9.541e-03 0.0%
09 0345 9.216e-03 0.0%
10 0384 9.174e-03 0.0%
11 0424 8.757e-03 0.0%
12 0463 8.948e-03 0.0%
13 0504 8.659e-03 0.0%
14 0544 8.602e-03 0.0%
15 0583 8.839e-03 0.0%
16 0621 8.513e-03 0.0%
17 0661 8.606e-03 0.0%
18 0702 8.344e-03 0.0%
19 0741 8.250e-03 0.0%
20 0781 8.467e-03 0.0%
21 0820 8.062e-03 0.0%
22 0860 8.203e-03 0.0%
23 0901 8.194e-03 0.0%
24 0941 7.995e-03 0.0%
25 0982 7.927e-03 0.0%
26 1022 8.018e-03 0.0%
27 1062 7.640e-03 0.0%
28 1101 7.695e-03 0.0%
29 1141 7.643e-03 0.0%
30 1182 7.841e-03 0.0%
31 1222 7.331e-03 0.0%
32 1262 7.633e-03 0.0%
33 1303 7.387e-03 0.0%
34 1344 7.604e-03 0.0%
35 1384 7.419e-03 0.0%
36 1424 7.501e-03 0.0%
37 1465 7.200e-03 0.0%
38 1505 7.354e-03 0.0%
39 1544 6.981e-03 0.0%
40 1584 6.742e-03 0.0%
41 1624 7.109e-03 0.0%
42 1664 6.919e-03 0.0%
43 1704 7.403e-03 0.0%
44 1743 6.916e-03 0.0%
45 1782 7.096e-03 0.0%
46 1822 6.828e-03 0.0%
47 1862 6.934e-03 0.0%
48 1902 6.900e-03 0.0%
49 1941 6.627e-03 0.0%
50 1980 6.938e-03 0.0%
51 2021 6.892e-03 0.0%
52 2061 6.962e-03 0.0%
53 2100 6.455e-03 0.0%
54 2139 6.573e-03 0.0%
55 2178 6.843e-03 0.0%
56 2218 6.696e-03 0.0%
57 2257 6.386e-03 0.0%
58 2298 6.595e-03 0.0%
59 2338 6.626e-03 0.0%
60 2378 6.162e-03 0.0%
61 2418 6.166e-03 0.0%
62 2458 6.382e-03 0.0%
63 2498 6.147e-03 0.0%
64 2539 6.230e-03 0.0%
65 2579 5.637e-03 0.0%
66 2619 6.042e-03 0.0%
67 2659 6.362e-03 0.0%
68 2699 5.919e-03 0.0%
69 2739 5.943e-03 0.0%
70 2779 6.310e-03 0.0%
71 2819 5.590e-03 0.0%
72 2861 6.093e-03 0.0%
73 2902 5.868e-03 0.0%
74 2943 5.497e-03 0.0%
75 2984 6.209e-03 0.0%
76 3025 5.460e-03 0.0%
77 3065 5.828e-03 0.0%
78 3105 5.765e-03 0.0%
79 3146 5.688e-03 0.0%
80 3187 5.164e-03 0.0%
81 3229 5.578e-03 0.0%
82 3270 5.443e-03 0.0%
83 3309 5.587e-03 0.0%
84 3349 5.402e-03 0.0%
85 3389 5.778e-03 0.0%
86 3429 5.515e-03 0.0%
87 3470 5.635e-03 0.0%
88 3511 5.444e-03 0.0%
89 3551 5.219e-03 0.0%
90 3592 5.381e-03 0.0%
91 3633 4.782e-03 0.0%
92 3674 4.935e-03 0.0%
93 3714 5.467e-03 0.0%
94 3754 5.389e-03 0.0%
95 3795 5.021e-03 0.0%
96 3835 4.834e-03 0.0%
97 3876 4.706e-03 0.0%
98 3917 5.298e-03 0.0%
99 3959 5.198e-03 0.0%
100 3999 4.741e-03 0.0%
101 4039 4.524e-03 0.0%
102 4079 4.572e-03 0.0%
103 4119 4.709e-03 0.0%
104 4159 4.860e-03 0.0%
105 4198 4.133e-03 0.0%
106 4238 4.894e-03 0.0%
107 4278 4.722e-03 0.0%
108 4318 4.106e-03 0.0%
109 4358 4.100e-03 0.0%
110 4398 4.954e-03 0.0%
111 4439 4.689e-03 0.0%
112 4480 4.248e-03 0.0%
113 4520 4.365e-03 0.0%
114 4561 4.721e-03 0.0%
115 4599 4.234e-03 0.0%
116 4639 4.063e-03 0.0%
117 4678 4.167e-03 0.0%
118 4718 3.808e-03 0.0%
119 4757 3.713e-03 0.0%
120 4795 4.195e-03 0.0%
121 4834 3.966e-03 0.0%
122 4873 3.812e-03 0.0%
123 4911 4.010e-03 0.0%
124 4951 4.235e-03 0.0%
125 4991 4.335e-03 0.0%
126 5031 4.211e-03 0.0%
127 5070 3.566e-03 0.0%
128 5108 3.651e-03 0.0%
129 5146 3.883e-03 0.0%
130 5185 3.872e-03 0.0%
131 5224 3.883e-03 0.0%
132 5265 3.831e-03 0.0%
133 5305 3.945e-03 0.0%
134 5345 3.751e-03 0.0%
135 5384 4.063e-03 0.0%
136 5423 3.910e-03 0.0%
137 5462 3.730e-03 0.0%
138 5502 3.301e-03 0.0%
xx 5518 train 0.0% test 0.0% (done)

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
    D: int = 256*8
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
        x = jax.vmap(norm)(jax.nn.relu(dln(x) + x))  # [L, D]
      return jax.nn.softmax(jax.vmap(self.final)(x))  # [L, Out]

  def get_batch(i):
    idx = jax.random.randint(rnd[i], shape=(10,), minval=0, maxval=wikitext2_train.shape[0]-1000)
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
