"""
train ComputeGate, an alternative to MLP

ComputeGate:
00 0000 6.407e+01 2.4%
01 0038 1.361e+01 19.1%
02 0077 7.560e+00 20.1%
03 0114 6.203e+00 21.7%
04 0151 4.751e+00 21.5%
05 0188 4.011e+00 25.1%
06 0225 3.441e+00 26.9%
07 0262 3.251e+00 28.1%
08 0299 2.927e+00 30.3%
09 0336 2.700e+00 33.2%
10 0373 2.507e+00 33.8%
11 0410 2.381e+00 35.2%
12 0447 2.290e+00 36.6%
13 0484 2.233e+00 37.3%
14 0521 2.196e+00 37.8%
15 0558 2.112e+00 39.0%
16 0595 2.122e+00 38.0%
17 0632 2.045e+00 40.8%
18 0669 2.019e+00 40.1%
19 0706 2.018e+00 41.5%
20 0743 1.967e+00 42.6%
21 0780 1.916e+00 43.2%
22 0817 1.885e+00 43.1%
23 0854 1.931e+00 44.1%
24 0891 1.893e+00 44.5%
25 0928 1.811e+00 45.0%
26 0965 1.814e+00 45.4%
27 1002 1.780e+00 45.0%
28 1039 1.820e+00 45.9%
29 1076 1.728e+00 46.4%
30 1113 1.728e+00 46.7%
31 1150 1.715e+00 47.0%
32 1187 1.705e+00 48.4%
33 1224 1.712e+00 47.9%
34 1261 1.630e+00 48.3%
35 1298 1.655e+00 49.3%
36 1335 1.664e+00 49.4%
37 1372 1.664e+00 49.0%
38 1409 1.606e+00 51.4%
39 1446 1.615e+00 50.9%
40 1483 1.528e+00 50.2%
41 1520 1.513e+00 52.4%
42 1557 1.553e+00 51.4%
43 1594 1.507e+00 52.6%
44 1631 1.481e+00 53.3%
45 1668 1.491e+00 53.0%
46 1705 1.479e+00 53.2%
47 1742 1.477e+00 53.2%
48 1779 1.477e+00 54.4%
49 1816 1.478e+00 54.4%
50 1853 1.480e+00 54.5%
51 1890 1.414e+00 54.9%
52 1927 1.416e+00 54.6%
53 1964 1.408e+00 55.9%
54 2001 1.389e+00 56.3%
55 2038 1.371e+00 57.0%
56 2075 1.438e+00 56.8%
57 2112 1.371e+00 56.8%
58 2149 1.396e+00 56.7%
59 2186 1.347e+00 57.9%
60 2223 1.372e+00 57.3%
61 2259 1.304e+00 58.1%
62 2296 1.321e+00 58.4%
63 2333 1.333e+00 58.5%
64 2370 1.331e+00 58.9%
65 2407 1.299e+00 59.5%
66 2444 1.202e+00 59.3%
67 2481 1.304e+00 58.9%
68 2518 1.252e+00 59.7%
69 2555 1.261e+00 59.9%
70 2592 1.229e+00 59.9%
71 2629 1.243e+00 60.7%
72 2666 1.237e+00 60.9%
73 2703 1.226e+00 60.4%
74 2740 1.212e+00 60.8%
75 2777 1.233e+00 60.7%
76 2814 1.209e+00 60.6%
77 2851 1.192e+00 61.8%
78 2888 1.175e+00 61.1%
79 2925 1.142e+00 62.1%
80 2962 1.192e+00 61.8%
81 2999 1.167e+00 61.9%
82 3036 1.113e+00 62.6%
83 3073 1.126e+00 62.4%
84 3110 1.131e+00 63.0%
85 3147 1.090e+00 62.1%
86 3184 1.089e+00 63.2%
87 3221 1.136e+00 62.7%
88 3258 1.077e+00 62.0%

MLP
starting
00 0000 6.314e+01 2.7%
01 0037 1.396e+01 18.1%
02 0075 8.118e+00 20.6%
03 0112 6.529e+00 19.5%
04 0150 4.879e+00 21.9%
05 0187 4.290e+00 21.3%
06 0224 3.654e+00 26.2%
07 0262 3.416e+00 28.4%
08 0300 2.889e+00 29.1%
09 0338 2.559e+00 33.7%
10 0376 2.342e+00 35.3%
11 0414 2.251e+00 37.8%
12 0452 2.102e+00 38.8%
13 0490 2.028e+00 41.0%
14 0528 1.990e+00 40.5%
15 0566 1.859e+00 42.0%
16 0603 1.918e+00 42.7%
17 0641 1.881e+00 43.7%
18 0679 1.821e+00 44.9%
19 0716 1.827e+00 45.0%
20 0753 1.781e+00 45.5%
21 0791 1.782e+00 46.5%
22 0829 1.726e+00 47.2%
23 0866 1.669e+00 47.9%
24 0904 1.686e+00 48.6%
25 0942 1.628e+00 48.7%
26 0980 1.652e+00 49.6%
27 1017 1.579e+00 50.3%
28 1055 1.575e+00 49.8%
29 1093 1.576e+00 50.7%
30 1130 1.545e+00 51.1%
31 1167 1.476e+00 51.7%
32 1204 1.533e+00 51.9%
33 1241 1.520e+00 52.4%
34 1278 1.473e+00 52.8%
35 1315 1.443e+00 52.9%
36 1352 1.460e+00 54.2%
37 1389 1.457e+00 54.2%
38 1426 1.384e+00 54.6%
39 1463 1.417e+00 54.7%
40 1500 1.414e+00 55.6%
41 1537 1.357e+00 55.5%
42 1574 1.344e+00 55.7%
43 1611 1.358e+00 56.7%
44 1648 1.338e+00 56.8%
45 1685 1.374e+00 57.0%
46 1722 1.337e+00 57.7%
47 1760 1.309e+00 57.9%
48 1797 1.282e+00 58.5%
49 1834 1.304e+00 58.5%
50 1872 1.295e+00 58.8%
51 1909 1.285e+00 59.0%
52 1946 1.266e+00 60.3%
53 1983 1.270e+00 60.4%
54 2021 1.236e+00 61.0%
55 2058 1.234e+00 61.4%
56 2095 1.164e+00 61.2%
57 2133 1.164e+00 61.7%
58 2170 1.241e+00 62.1%
59 2207 1.142e+00 61.9%
60 2244 1.214e+00 61.5%
61 2282 1.164e+00 62.6%
62 2319 1.170e+00 63.0%
63 2356 1.136e+00 63.0%
64 2393 1.152e+00 62.1%
65 2430 1.176e+00 63.2%
66 2467 1.144e+00 63.8%
67 2504 1.108e+00 63.8%
68 2541 1.100e+00 64.3%
69 2578 1.107e+00 64.1%
70 2615 1.095e+00 64.7%
71 2652 1.078e+00 64.8%
72 2689 1.054e+00 64.9%
73 2726 1.081e+00 64.5%
74 2763 1.083e+00 64.9%
75 2800 1.068e+00 65.0%
76 2837 1.009e+00 64.8%
77 2874 1.049e+00 66.1%
78 2911 1.071e+00 65.5%
79 2949 1.009e+00 66.5%
80 2986 1.029e+00 66.3%
81 3023 9.794e-01 66.9%
82 3060 9.981e-01 66.8%
83 3097 1.030e+00 66.7%
84 3134 9.919e-01 67.0%
85 3171 1.025e+00 66.0%
86 3208 9.051e-01 66.8%
87 3245 9.908e-01 67.0%
88 3282 9.863e-01 67.8%

"""


def main():

    @funtree
    def Mlp(x, key, up, down, dropout_p: float):
        x_norm = rms_norm(x)
        expanded = jax.nn.gelu(einsum(x_norm, up, 'L E, E U -> L U'))
        lowered = einsum(expanded, down, 'L U, U E -> L E')
        return dropout(lowered, key, dropout_p)

    # activation: id < gelu << sigmoid
    @funtree
    def ComputeGate(x, key, gate_m, compute_m, down, expansion: int, dropout_p: float):
        x_norm = rms_norm(x)
        gate = jax.nn.sigmoid(einsum(x_norm, gate_m, 'L E, E U -> L U'))
        compute = einsum(x_norm, compute_m, 'L E, E U -> L U')
        # pooling is way worse: merged = reduce(gate * compute, 'L (E n) -> L E', 'max', n=expansion)
        merged = einsum(gate * compute, down, 'L U, U E -> L E')
        return dropout(merged, key, dropout_p)

    @ funtree
    def Attention(x, key, qkv, out, heads: int, dropout_p: float):
        x_norm = rms_norm(x)
        parts = einsum(x_norm, qkv, 'L E, E HsplitD -> L HsplitD')
        k, q, v = rearrange(parts, 'L (H split D) -> split H L D', split=3, H=heads)
        q, k = norm(q), norm(k)
        L, D = x.shape
        mask = jnp.tril(jnp.ones([L, L]))
        similarity = einsum(k, q, 'H L D, H L2 D -> H L L2') * (D ** -0.5)
        masked_similarity = jnp.where(mask, similarity, -jnp.inf)
        attention = jax.nn.softmax(masked_similarity, axis=-1)
        attention = dropout(attention, key, dropout_p)
        gather = einsum(attention, v, 'H L L2, H L2 V -> H L V')
        gather = rearrange(gather, 'H L V -> L (H V)')
        output = einsum(gather, out, 'L Z, Z E -> L E')
        return output

    @ funtree
    def GPT(x, key, embedding, positional, layers, unembed):
        L = x.shape[0]
        hidden = embedding[x] + positional[:L, :]
        for layer, k in zipkey(layers, key):
            hidden = hidden + layer(hidden, key=k)
        logits = einsum(unembed, hidden, 'E O, L E -> L O')
        return logits

    def zipkey(items, key):
        return zip(items, jax.random.split(key, len(items)))

    def init_gpt_model(vocab, embedding, heads, layer_count, expansion, max_length):
        init = Initializer(jax.random.PRNGKey(0))
        def make_attention(init):
            return Attention(
                qkv=init.glorot_normal([embedding, 3*heads*embedding]),
                out=init.glorot_normal([heads*embedding, embedding]),
                heads=heads,
                dropout_p=0.1,
            )
        def make_mlp(init):
            return ComputeGate(
                gate_m=init.glorot_normal([embedding, expansion*embedding]),
                compute_m=init.glorot_normal([embedding, expansion*embedding]),
                down=init.glorot_normal([expansion*embedding, embedding]),
                expansion=expansion,
                dropout_p=0.1
            )
            # return Mlp(
            #    up=init.glorot_normal([embedding, expansion*embedding]),
            #    down=init.glorot_normal([expansion*embedding, embedding]),
            #    dropout_p=0.1,
            # )
        return GPT(
            embedding=init.normal([vocab, embedding], 1),
            positional=init.normal([max_length, embedding], 1),
            layers=init.map([make_attention, make_mlp]*layer_count),
            unembed=init.normal([embedding, vocab], 1),
        )

    data = utils.run_const(lambda: jnp.load('shakespeare_char/train.npy'))
    meta = json.load(open('shakespeare_char/meta.json'))
    char_map = jnp.array([ord(c) for c in meta['chars']])
    newline = [i for i, v in enumerate(meta['chars']) if v == '\n'][0]
    break_positions = jnp.where(jnp.logical_and(data[:-1] == newline, data[1:] == newline))[0] + 2

    def as_text(vals):
        return ''.join(chr(c) for c in char_map[vals]).replace('\n', '¶')

    def tasks(key):
        start = jr.choice(key, break_positions, shape=[batch_size])[:, None]
        offset = jnp.arange(seq_length)[None, :]
        return (data[start + offset], data[start + offset + 1])

    def accuracy_fn(batch):  # ([B, L], [B, L])
        x, y = batch
        def fn(model, key):
            keys = jr.split(key, x.shape[0])
            logits = jax.vmap(model)(x, keys)
            prediction = jnp.argmax(logits, axis=-1)  # argmax(softmax(x)) == argmax(x) - softmax preserves max
            return jnp.mean(prediction == y, axis=(0, 1))
        return jax.jit(fn)

    def loss_fn(model, x, y, key):  # model, [B, L], [B, L]
        logits = jax.vmap(model)(x, jr.split(key, x.shape[0]))  # [B, L, C]
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    def update(key, model, opt_state):
        task_key, model_key = jr.split(key)
        x, y = tasks(task_key)
        return update_with_task(x, y, model, opt_state, model_key)

    @ jax.jit
    def update_with_task(x, y, model, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return loss, model, opt_state

    batch_size = 64
    seq_length = 128
    model = init_gpt_model(vocab=39, embedding=256, heads=6, layer_count=6, expansion=4, max_length=seq_length)
    decay_mask = funmap(model, {
        Attention: lambda **kw: Update(kw, qkv=True, out=True, key=False),
        Mlp: lambda **kw: Update(kw, up=True, down=True, key=False),
        ComputeGate: lambda **kw: Update(kw, gate_m=True, compute_m=True, down=True, key=False),
        GPT: lambda **kw: Update(kw, embedding=False, positional=False, unembed=False, key=False),
    })
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0, peak_value=3e-4, warmup_steps=100, decay_steps=500, end_value=1e-4)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(0.9, 0.99),
        optax.add_decayed_weights(0.1, lambda x: decay_mask),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1)
    )
    opt_state = optimizer.init(model)
    accuracy_set = tasks(key=jax.random.PRNGKey(-1))

    mode = 'LEARN'

    if mode == 'LEARN':
        return utils.optimize(model, opt_state, update, accuracy_fn(accuracy_set))
    elif mode == 'PRED':
        for a in range(25):
            k = jax.random.PRNGKey(a)
            task = tasks(k)[0][0]
            print(as_text(task))
            print(' ' + as_text(jnp.argmax(final_model(task, k), axis=-1)))
    elif mode == 'GEN':
        x = jnp.array([15])
        k = jax.random.PRNGKey(0)
        for i in range(70):
            next = jnp.argmax(final_model(x, k)[-1:, :], axis=-1)
            x = jnp.concatenate([x, next])
        print(as_text(x))
    return final_model, final_opt_state, final_data
