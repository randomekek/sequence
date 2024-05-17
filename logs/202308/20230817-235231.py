"""
try DLN using character level 256 way softmax

00 0000 0.02 0.0%
01 0009 0.02 0.0%
02 0023 0.01 0.0%
03 0037 0.01 0.0%
04 0050 0.01 0.0%
05 0064 0.01 0.0%
06 0078 0.01 0.0%
07 0092 0.01 0.0%
08 0106 0.01 0.0%
09 0120 0.01 0.0%
10 0134 0.01 0.0%
11 0148 0.01 0.0%
12 0162 0.01 0.0%
13 0176 0.01 0.0%
14 0190 0.01 0.0%
15 0204 0.01 0.0%
16 0218 0.01 0.0%
17 0232 0.01 0.0%
18 0246 0.01 0.0%
19 0260 0.01 0.0%
20 0274 0.01 0.0%
21 0288 0.01 0.0%
22 0302 0.01 0.0%
23 0316 0.01 0.0%
24 0330 0.01 0.0%
25 0344 0.01 0.0%
26 0358 0.01 0.0%
27 0372 0.01 0.0%
28 0386 0.01 0.0%
29 0400 0.01 0.0%
30 0413 0.01 0.0%
31 0426 0.01 0.0%
32 0439 0.01 0.0%
33 0453 0.01 0.0%
34 0467 0.01 0.0%
35 0481 0.01 0.0%
36 0495 0.01 0.0%
37 0508 0.01 0.0%
38 0522 0.01 0.0%
39 0536 0.01 0.0%
40 0550 0.01 0.0%
41 0564 0.01 0.0%
42 0578 0.01 0.0%
43 0592 0.01 0.0%
44 0606 0.01 0.0%
45 0620 0.01 0.0%
46 0634 0.01 0.0%
47 0648 0.01 0.0%
48 0662 0.01 0.0%
49 0676 0.01 0.0%
50 0690 0.01 0.0%
51 0704 0.01 0.0%
52 0718 0.01 0.0%
53 0732 0.01 0.0%
54 0746 0.01 0.0%
55 0760 0.01 0.0%
56 0774 0.01 0.0%
57 0788 0.01 0.0%
58 0802 0.01 0.0%
59 0816 0.01 0.0%
60 0830 0.01 0.0%
61 0844 0.01 0.0%
62 0858 0.01 0.0%
63 0872 0.01 0.0%
64 0886 0.01 0.0%
65 0900 0.01 0.0%
66 0914 0.01 0.0%
67 0928 0.01 0.0%
68 0942 0.01 0.0%
69 0956 0.01 0.0%
70 0970 0.01 0.0%
71 0984 0.01 0.0%
72 0998 0.01 0.0%
73 1012 0.01 0.0%
74 1026 0.01 0.0%
75 1039 0.01 0.0%
76 1053 0.01 0.0%
77 1067 0.01 0.0%
78 1081 0.01 0.0%
79 1095 0.01 0.0%
80 1109 0.01 0.0%
81 1123 0.01 0.0%
82 1137 0.01 0.0%
83 1151 0.01 0.0%
84 1165 0.01 0.0%
85 1179 0.01 0.0%
86 1193 0.01 0.0%
87 1207 0.01 0.0%
88 1221 0.01 0.0%
89 1236 0.01 0.0%
90 1250 0.01 0.0%
91 1264 0.01 0.0%
92 1279 0.01 0.0%
93 1293 0.01 0.0%
94 1307 0.01 0.0%
95 1321 0.01 0.0%
96 1335 0.01 0.0%
97 1349 0.01 0.0%
98 1363 0.01 0.0%
99 1378 0.01 0.0%
100 1393 0.01 0.0%
101 1408 0.01 0.0%
102 1423 0.01 0.0%
103 1438 0.01 0.0%
104 1453 0.01 0.0%
105 1468 0.01 0.0%
106 1482 0.01 0.0%
107 1497 0.01 0.0%
108 1512 0.01 0.0%
109 1526 0.01 0.0%
110 1540 0.01 0.0%
111 1554 0.01 0.0%
112 1568 0.01 0.0%
113 1582 0.01 0.0%
114 1596 0.01 0.0%
115 1610 0.01 0.0%
116 1624 0.01 0.0%
117 1638 0.01 0.0%
118 1652 0.01 0.0%
119 1666 0.01 0.0%
120 1680 0.01 0.0%
121 1694 0.01 0.0%
122 1708 0.01 0.0%
123 1722 0.01 0.0%
124 1736 0.01 0.0%
125 1750 0.01 0.0%
126 1764 0.01 0.0%
127 1778 0.01 0.0%
128 1791 0.01 0.0%
129 1805 0.01 0.0%
130 1818 0.01 0.0%
131 1831 0.01 0.0%
132 1845 0.01 0.0%
133 1858 0.01 0.0%
134 1871 0.01 0.0%
135 1884 0.01 0.0%
136 1897 0.01 0.0%
137 1911 0.01 0.0%
138 1925 0.01 0.0%
139 1938 0.01 0.0%
140 1951 0.01 0.0%
141 1964 0.01 0.0%
142 1977 0.01 0.0%
143 1990 0.01 0.0%
144 2003 0.01 0.0%
145 2016 0.01 0.0%
146 2029 0.01 0.0%
147 2043 0.01 0.0%
148 2057 0.01 0.0%
149 2071 0.01 0.0%
150 2085 0.01 0.0%
151 2099 0.01 0.0%
152 2112 0.01 0.0%
153 2125 0.01 0.0%
154 2138 0.01 0.0%
155 2151 0.01 0.0%
156 2164 0.01 0.0%
157 2177 0.01 0.0%
158 2190 0.01 0.0%
159 2203 0.01 0.0%
160 2217 0.01 0.0%
161 2231 0.01 0.0%
162 2245 0.01 0.0%
163 2259 0.01 0.0%
164 2273 0.01 0.0%
165 2287 0.01 0.0%
166 2301 0.01 0.0%
167 2314 0.01 0.0%
168 2328 0.01 0.0%
169 2342 0.01 0.0%
170 2356 0.01 0.0%
171 2370 0.01 0.0%
172 2384 0.01 0.0%
173 2398 0.01 0.0%
174 2412 0.01 0.0%
175 2425 0.00 0.0%
176 2438 0.01 0.0%
177 2452 0.01 0.0%
178 2466 0.00 0.0%
179 2479 0.00 0.0%
180 2493 0.00 0.0%
181 2507 0.01 0.0%
182 2521 0.01 0.0%
183 2535 0.00 0.0%
184 2549 0.00 0.0%
185 2563 0.00 0.0%
186 2577 0.00 0.0%
187 2591 0.00 0.0%
188 2604 0.00 0.0%
189 2618 0.00 0.0%
190 2632 0.00 0.0%
191 2646 0.00 0.0%
192 2660 0.00 0.0%
193 2674 0.00 0.0%
194 2687 0.00 0.0%
195 2701 0.00 0.0%
196 2715 0.00 0.0%
197 2729 0.00 0.0%
198 2743 0.00 0.0%
199 2757 0.00 0.0%
200 2770 0.00 0.0%
201 2783 0.00 0.0%
202 2796 0.00 0.0%
203 2809 0.00 0.0%
204 2822 0.00 0.0%
205 2835 0.00 0.0%
206 2848 0.00 0.0%
207 2861 0.00 0.0%
208 2874 0.00 0.0%
209 2887 0.00 0.0%
210 2901 0.00 0.0%
211 2915 0.00 0.0%
212 2929 0.00 0.0%
213 2942 0.00 0.0%
214 2955 0.00 0.0%
215 2969 0.00 0.0%
216 2983 0.00 0.0%
217 2997 0.00 0.0%
218 3011 0.00 0.0%
219 3024 0.00 0.0%
220 3037 0.00 0.0%
221 3051 0.00 0.0%
222 3065 0.00 0.0%
223 3078 0.00 0.0%
224 3091 0.00 0.0%
225 3104 0.00 0.0%
226 3117 0.00 0.0%
227 3131 0.00 0.0%
228 3144 0.00 0.0%
229 3157 0.00 0.0%
230 3170 0.00 0.0%
231 3184 0.00 0.0%
232 3198 0.00 0.0%
233 3212 0.00 0.0%
234 3226 0.00 0.0%
235 3239 0.00 0.0%
236 3253 0.00 0.0%
237 3267 0.00 0.0%
238 3281 0.00 0.0%
239 3295 0.00 0.0%
240 3309 0.00 0.0%
241 3323 0.00 0.0%
242 3337 0.00 0.0%
243 3351 0.00 0.0%
244 3364 0.00 0.0%
245 3377 0.00 0.0%
246 3390 0.00 0.0%
247 3404 0.00 0.0%
248 3418 0.00 0.0%
249 3432 0.00 0.0%
250 3445 0.00 0.0%
251 3459 0.00 0.0%
252 3473 0.00 0.0%
253 3487 0.00 0.0%
254 3501 0.00 0.0%
255 3515 0.00 0.0%
256 3529 0.00 0.0%
257 3543 0.00 0.0%
258 3557 0.00 0.0%
259 3571 0.00 0.0%
260 3584 0.00 0.0%
261 3598 0.00 0.0%
262 3611 0.00 0.0%
263 3624 0.00 0.0%
264 3637 0.00 0.0%
265 3650 0.00 0.0%
266 3663 0.00 0.0%
267 3676 0.00 0.0%
268 3689 0.00 0.0%
269 3702 0.00 0.0%
270 3715 0.00 0.0%
271 3728 0.00 0.0%
272 3742 0.00 0.0%
273 3755 0.00 0.0%
274 3768 0.00 0.0%
275 3781 0.00 0.0%
276 3794 0.00 0.0%
xx 3798 train 0.0% test 0.0% (done)

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
    idx = jax.random.randint(rnd[i], shape=(30,), minval=0, maxval=wikitext2_train.shape[0]-1000)
    x = jnp.array([wikitext2_train[i:i+1000] for i in idx])
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
