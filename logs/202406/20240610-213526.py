"""
compare gelu, relu, sigmoid and tanh
"""

def main():
    from einops import einsum, rearrange, reduce
    from funtree import dropout, norm, rms_norm
    import funtree
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import json
    import optax
    import utils

    @funtree.makefun
    def MLP(x, key, up, down, base_scale: float, bump: float, dropout_p: float, act: str):
        x_norm = jax.vmap(rms_norm)(x)
        activation = {
            'gelu': jax.nn.gelu,
            'relu': jax.nn.relu,
            'sigmoid': lambda x: x - base_scale * (jax.nn.sigmoid(x) - 0.5),
            'tanh': lambda x: x - bump * base_scale * jax.nn.tanh(1. / base_scale * x),
        }[act]
        expanded = activation(einsum(x_norm, up, 'L E, E U -> L U'))
        lowered = einsum(expanded, down, 'L U, U E -> L E')
        return dropout(lowered, key, dropout_p)

    @funtree.makefun
    def Attention(x, key, qk_proj, v_proj, out, heads: int, dropout_p: float):
        x_norm = jax.vmap(rms_norm)(x)
        parts = einsum(x_norm, qk_proj, 'L E, E splitHD -> L splitHD')
        k, q = rearrange(parts, 'L (split H D) -> split H L D', split=2, H=heads)
        q, k = jax.vmap(jax.vmap(norm))(q), jax.vmap(jax.vmap(norm))(k)
        H, L, D = k.shape
        v = rearrange(einsum(x_norm, v_proj, 'L E, E HD -> L HD'), 'L (H D) -> H L D', H=heads)
        mask = jnp.tril(jnp.ones([L, L]))
        similarity = einsum(k, q, 'H L D, H L2 D -> H L L2')
        masked_similarity = jnp.where(mask, similarity, -jnp.inf)
        attention = jax.nn.softmax(masked_similarity.astype(jnp.float32) / jnp.sqrt(D), axis=-1)
        attention = attention.astype(jnp.bfloat16)
        key1, key2 = jr.split(key)
        attention = dropout(attention, key1, dropout_p)
        fetch = einsum(attention, v, 'H L L2, H L2 V -> H L V')
        gather = rearrange(fetch, 'H L V -> L (H V)')
        output = einsum(gather, out, 'L Z, Z E -> L E')
        output = dropout(output, key2, dropout_p)
        return output

    @funtree.makefun
    def Block(x, key, attn, mlp):
        attn_key, mlp_key = jr.split(key)
        x = x + attn(x, key=attn_key)
        x = x + mlp(x, key=mlp_key)
        return x

    @funtree.makefun
    def GPT(x, key, embedding, positional, layers, unembed):
        L = x.shape[0]
        hidden = embedding[x] + positional[:L, :]
        for layer, k in funtree.zipkey(layers, key):
            hidden = hidden + layer(hidden, key=k)
        logits = einsum(unembed, hidden, 'E O, L E -> L O')
        return logits

    def init_gpt(seq_length, layer_count, embed_size, heads, vocab, dropout_p, qk_scale, emb_scale, v_scale, base_scale, bump, act):
        init = funtree.Initializer(jr.PRNGKey(0))
        def make_layer(init: funtree.Initializer):
            return Block(
                attn=Attention(
                    qk_proj=init.glorot_normal([embed_size, embed_size * 2]) * qk_scale,
                    v_proj=init.glorot_normal([embed_size, embed_size]) * v_scale,
                    out=init.glorot_normal([embed_size, embed_size]),
                    heads=heads,
                    dropout_p=dropout_p),
                mlp=MLP(
                    up=init.glorot_normal([embed_size, 4 * embed_size]),
                    down=init.glorot_normal([4 * embed_size, embed_size]),
                    base_scale=base_scale,
                    bump=bump,
                    dropout_p=dropout_p,
                    act=act))
        embedding = emb_scale * init.normal([vocab, embed_size]) * jax.lax.rsqrt(1. * embed_size)
        return GPT(
            embedding=embedding,
            positional=init.glorot_normal([seq_length, embed_size]),
            layers=init.map([make_layer] * layer_count),
            unembed=embedding.T)

    data = utils.run_const(lambda: jnp.load('shakespeare_char/train.npy'))
    meta = json.load(open('shakespeare_char/meta.json'))
    newline = [i for i, v in enumerate(meta['chars']) if v == '\n'][0]
    break_positions = jnp.where(jnp.logical_and(data[:-1] == newline, data[1:] == newline))[0] + 2

    def tasks(key):
        start = jr.choice(key, break_positions, shape=[batch_size])[:, None]
        offset = jnp.arange(seq_length)[None, :]
        return (data[start + offset], data[start + offset + 1])

    def loss_fn(model, x, y, key):  # model, [B, L], [B, L]
        logits = jax.vmap(model)(x, jr.split(key, x.shape[0]))  # [B, L, C]
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    def update(key, model, opt_state):
        task_key, model_key = jr.split(key)
        x, y = tasks(task_key)
        return update_with_task(x, y, model, opt_state, model_key)

    @jax.jit
    def update_with_task(x, y, model, opt_state, key):
        model_bfloat16 = utils.cast_pytree(model, jnp.bfloat16)
        loss, grads = jax.value_and_grad(loss_fn)(model_bfloat16, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return loss, model, opt_state

    batch_size = 64
    seq_length = 256
    base_params = dict(seq_length=seq_length, layer_count=6, embed_size=384, heads=6,
                       vocab=65, dropout_p=0.2, qk_scale=0.1, emb_scale=0.03, v_scale=0.01)
    models = {
        f'gelu': init_gpt(**base_params, base_scale=None, bump=None, act='gelu'),
        f'relu': init_gpt(**base_params, base_scale=None, bump=None, act='relu'),
        f'sigmoid': init_gpt(**base_params, base_scale=4.0, bump=None, act='sigmoid'),
        f'tanh': init_gpt(**base_params, base_scale=3.0, bump=1.05, act='tanh'),
    }
    outputs = {}
    for name, model in models.items():
        print(f'config: {name}')
        decay_weights = funtree.funmap(model, {
            Block: lambda **kw: kw,
            Attention: lambda **kw: kw | dict(qk_proj=True, v_proj=True, out=True),
            MLP: lambda **kw: kw | dict(up=True, down=True),
            GPT: lambda **kw: kw | dict(embedding=False, positional=False, unembed=False),
        })
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(0.9, 0.99),
            optax.add_decayed_weights(0.1, lambda x: decay_weights),
            optax.scale_by_schedule(optax.warmup_cosine_decay_schedule(
                init_value=0,
                peak_value=1e-3,
                warmup_steps=100,
                decay_steps=5000,
                end_value=1e-4)),
            optax.scale(-1))
        opt_state = optimizer.init(model)
        model, opt_state, interrupted = utils.optimize(model, opt_state, update, iter_count=4000)
        outputs[name] = dict(model=model, opt_state=opt_state)
        if interrupted:
            break
    return outputs


"""
config: gelu
00 0000 4.406e+00 0.0it/s 
01 0039 2.703e+00 7.6it/s 
02 0078 2.516e+00 7.6it/s 
03 0117 2.281e+00 7.6it/s 
04 0156 2.125e+00 7.7it/s 
05 0195 2.047e+00 7.7it/s 
06 0234 1.922e+00 7.6it/s 
07 0273 1.836e+00 7.6it/s 
08 0311 1.719e+00 7.6it/s 
09 0349 1.641e+00 7.5it/s 
10 0387 1.641e+00 7.6it/s 
11 0425 1.617e+00 7.6it/s 
12 0463 1.539e+00 7.6it/s 
13 0501 1.469e+00 7.6it/s 
14 0539 1.508e+00 7.6it/s 
15 0577 1.453e+00 7.6it/s 
16 0615 1.406e+00 7.5it/s 
17 0653 1.406e+00 7.6it/s 
18 0691 1.430e+00 7.6it/s 
19 0729 1.367e+00 7.6it/s 
20 0767 1.383e+00 7.5it/s 
21 0805 1.336e+00 7.5it/s 
22 0843 1.328e+00 7.6it/s 
23 0881 1.258e+00 7.5it/s 
24 0919 1.266e+00 7.5it/s 
25 0957 1.289e+00 7.6it/s 
26 0995 1.305e+00 7.6it/s 
27 1033 1.289e+00 7.6it/s 
28 1071 1.234e+00 7.5it/s 
29 1109 1.250e+00 7.6it/s 
30 1147 1.156e+00 7.5it/s 
31 1185 1.180e+00 7.5it/s 
32 1223 1.242e+00 7.5it/s 
33 1262 1.164e+00 7.6it/s 
34 1301 1.195e+00 7.6it/s 
35 1339 1.117e+00 7.5it/s 
36 1377 1.102e+00 7.5it/s 
37 1415 1.109e+00 7.5it/s 
38 1453 1.102e+00 7.5it/s 
39 1491 1.125e+00 7.5it/s 
40 1529 1.133e+00 7.5it/s 
41 1567 1.062e+00 7.5it/s 
42 1605 1.039e+00 7.5it/s 
43 1643 1.102e+00 7.5it/s 
44 1681 1.008e+00 7.5it/s 
45 1719 1.039e+00 7.5it/s 
46 1757 1.047e+00 7.5it/s 
47 1795 9.414e-01 7.5it/s 
48 1833 9.570e-01 7.6it/s 
49 1871 1.023e+00 7.5it/s 
50 1909 9.961e-01 7.6it/s 
51 1947 9.805e-01 7.6it/s 
52 1985 9.219e-01 7.5it/s 
53 2023 9.258e-01 7.5it/s 
54 2061 9.023e-01 7.6it/s 
55 2099 8.945e-01 7.6it/s 
56 2137 8.984e-01 7.5it/s 
57 2175 9.102e-01 7.5it/s 
58 2213 8.633e-01 7.6it/s 
59 2251 8.398e-01 7.6it/s 
60 2289 9.062e-01 7.5it/s 
61 2327 8.281e-01 7.6it/s 
62 2365 7.852e-01 7.6it/s 
63 2403 8.008e-01 7.6it/s 
64 2441 8.281e-01 7.5it/s 
65 2479 7.695e-01 7.6it/s 
66 2517 7.617e-01 7.6it/s 
67 2555 7.852e-01 7.5it/s 
68 2593 7.109e-01 7.5it/s 
69 2631 7.227e-01 7.6it/s 
70 2669 8.242e-01 7.6it/s 
71 2707 7.539e-01 7.5it/s 
72 2745 7.109e-01 7.6it/s 
73 2783 6.992e-01 7.6it/s 
74 2821 6.875e-01 7.6it/s 
75 2859 6.836e-01 7.5it/s 
76 2897 6.562e-01 7.6it/s 
77 2935 7.070e-01 7.6it/s 
78 2973 6.641e-01 7.5it/s 
79 3011 6.289e-01 7.5it/s 
80 3049 5.742e-01 7.6it/s 
81 3087 6.250e-01 7.6it/s 
82 3125 5.977e-01 7.5it/s 
83 3163 6.523e-01 7.6it/s 
84 3201 6.289e-01 7.6it/s 
85 3239 6.211e-01 7.6it/s 
86 3277 5.820e-01 7.5it/s 
87 3315 5.977e-01 7.6it/s 
88 3353 6.133e-01 7.6it/s 
89 3391 5.469e-01 7.5it/s 
90 3429 5.781e-01 7.5it/s 
91 3467 5.000e-01 7.6it/s 
92 3505 5.195e-01 7.6it/s 
93 3543 5.781e-01 7.5it/s 
94 3581 5.234e-01 7.6it/s 
95 3619 5.391e-01 7.6it/s 
96 3657 4.883e-01 7.6it/s 
97 3695 4.414e-01 7.5it/s 
98 3733 5.547e-01 7.6it/s 
99 3771 5.273e-01 7.6it/s 
100 3809 4.922e-01 7.5it/s 
101 3847 5.156e-01 7.5it/s 
102 3885 4.375e-01 7.6it/s 
103 3923 5.273e-01 7.6it/s 
104 3961 5.312e-01 7.5it/s 
105 3999 4.844e-01 7.5it/s 
xx 3999 4.590e-01  (done)
config: relu
00 0000 4.500e+00 0.0it/s 
01 0040 2.688e+00 7.9it/s 
02 0079 2.516e+00 7.7it/s 
03 0119 2.328e+00 7.9it/s 
04 0159 2.156e+00 7.8it/s 
05 0198 2.047e+00 7.7it/s 
06 0237 1.969e+00 7.7it/s 
07 0276 1.867e+00 7.7it/s 
08 0315 1.750e+00 7.7it/s 
09 0354 1.703e+00 7.7it/s 
10 0393 1.602e+00 7.7it/s 
11 0432 1.570e+00 7.7it/s 
12 0471 1.562e+00 7.7it/s 
13 0510 1.477e+00 7.7it/s 
14 0549 1.508e+00 7.7it/s 
15 0588 1.477e+00 7.7it/s 
16 0627 1.453e+00 7.7it/s 
17 0666 1.383e+00 7.7it/s 
18 0705 1.367e+00 7.7it/s 
19 0744 1.328e+00 7.7it/s 
20 0783 1.336e+00 7.7it/s 
21 0822 1.352e+00 7.7it/s 
22 0861 1.266e+00 7.6it/s 
23 0900 1.281e+00 7.6it/s 
24 0939 1.258e+00 7.7it/s 
25 0978 1.297e+00 7.8it/s 
26 1017 1.258e+00 7.8it/s 
27 1056 1.219e+00 7.8it/s 
28 1095 1.227e+00 7.7it/s 
29 1134 1.242e+00 7.7it/s 
30 1173 1.188e+00 7.6it/s 
31 1211 1.180e+00 7.5it/s 
32 1250 1.172e+00 7.6it/s 
33 1289 1.203e+00 7.8it/s 
34 1328 1.141e+00 7.6it/s 
35 1366 1.102e+00 7.5it/s 
36 1404 1.172e+00 7.6it/s 
37 1443 1.148e+00 7.7it/s 
38 1482 1.172e+00 7.7it/s 
39 1521 1.078e+00 7.7it/s 
40 1560 1.062e+00 7.8it/s 
41 1599 1.070e+00 7.8it/s 
42 1638 1.062e+00 7.6it/s 
43 1677 1.070e+00 7.7it/s 
44 1716 1.062e+00 7.8it/s 
45 1755 1.031e+00 7.6it/s 
46 1794 1.031e+00 7.7it/s 
47 1833 9.805e-01 7.7it/s 
48 1872 1.000e+00 7.7it/s 
49 1911 9.688e-01 7.7it/s 
50 1950 9.961e-01 7.6it/s 
51 1989 9.531e-01 7.7it/s 
52 2028 9.453e-01 7.7it/s 
53 2067 9.297e-01 7.7it/s 
54 2106 9.375e-01 7.6it/s 
55 2145 9.180e-01 7.7it/s 
56 2184 9.023e-01 7.6it/s 
57 2223 9.102e-01 7.7it/s 
58 2262 8.711e-01 7.6it/s 
59 2301 8.164e-01 7.7it/s 
60 2340 8.516e-01 7.7it/s 
61 2379 8.750e-01 7.7it/s 
62 2418 8.203e-01 7.7it/s 
63 2457 8.945e-01 7.7it/s 
64 2496 8.320e-01 7.7it/s 
65 2535 7.734e-01 7.7it/s 
66 2574 8.359e-01 7.7it/s 
67 2613 8.203e-01 7.7it/s 
68 2652 7.930e-01 7.6it/s 
69 2691 7.891e-01 7.6it/s 
70 2730 7.812e-01 7.6it/s 
71 2768 8.516e-01 7.6it/s 
72 2807 6.992e-01 7.6it/s 
73 2846 7.227e-01 7.7it/s 
74 2885 7.734e-01 7.6it/s 
75 2924 7.188e-01 7.6it/s 
76 2963 7.148e-01 7.7it/s 
77 3002 7.422e-01 7.7it/s 
78 3041 6.719e-01 7.6it/s 
79 3079 6.680e-01 7.6it/s 
80 3118 6.836e-01 7.7it/s 
81 3157 6.992e-01 7.6it/s 
82 3195 6.719e-01 7.5it/s 
83 3234 6.484e-01 7.6it/s 
84 3273 6.914e-01 7.7it/s 
85 3312 6.445e-01 7.6it/s 
86 3351 6.328e-01 7.7it/s 
87 3390 6.289e-01 7.7it/s 
88 3429 6.289e-01 7.7it/s 
89 3468 5.859e-01 7.7it/s 
90 3507 5.625e-01 7.7it/s 
91 3546 6.484e-01 7.7it/s 
92 3585 6.406e-01 7.7it/s 
93 3624 6.133e-01 7.7it/s 
94 3663 5.859e-01 7.7it/s 
95 3702 5.352e-01 7.7it/s 
96 3741 5.703e-01 7.7it/s 
97 3780 5.586e-01 7.7it/s 
98 3819 5.469e-01 7.8it/s 
99 3859 5.312e-01 7.8it/s 
100 3898 5.977e-01 7.8it/s 
101 3937 5.273e-01 7.8it/s 
102 3976 5.195e-01 7.8it/s 
xx 3999 5.312e-01  (done)
config: sigmoid
00 0000 4.188e+00 0.0it/s 
01 0040 3.062e+00 7.9it/s 
02 0079 2.516e+00 7.8it/s 
03 0118 2.453e+00 7.7it/s 
04 0157 2.344e+00 7.7it/s 
05 0196 2.109e+00 7.7it/s 
06 0235 1.992e+00 7.7it/s 
07 0274 1.922e+00 7.6it/s 
08 0313 1.836e+00 7.6it/s 
09 0352 1.758e+00 7.6it/s 
10 0391 1.680e+00 7.6it/s 
11 0429 1.641e+00 7.6it/s 
12 0468 1.555e+00 7.6it/s 
13 0507 1.539e+00 7.6it/s 
14 0546 1.484e+00 7.6it/s 
15 0584 1.500e+00 7.6it/s 
16 0623 1.430e+00 7.6it/s 
17 0662 1.375e+00 7.6it/s 
18 0700 1.336e+00 7.6it/s 
19 0739 1.359e+00 7.6it/s 
20 0778 1.320e+00 7.6it/s 
21 0816 1.266e+00 7.6it/s 
22 0854 1.297e+00 7.6it/s 
23 0893 1.258e+00 7.6it/s 
24 0932 1.211e+00 7.6it/s 
25 0970 1.250e+00 7.6it/s 
26 1009 1.227e+00 7.6it/s 
27 1048 1.188e+00 7.6it/s 
28 1086 1.211e+00 7.6it/s 
29 1124 1.180e+00 7.6it/s 
30 1163 1.211e+00 7.6it/s 
31 1202 1.195e+00 7.6it/s 
32 1240 1.148e+00 7.6it/s 
33 1278 1.141e+00 7.6it/s 
34 1317 1.172e+00 7.6it/s 
35 1355 1.117e+00 7.6it/s 
36 1393 1.102e+00 7.6it/s 
37 1432 1.094e+00 7.6it/s 
38 1471 1.070e+00 7.6it/s 
39 1509 1.039e+00 7.6it/s 
40 1547 1.055e+00 7.6it/s 
41 1586 1.016e+00 7.6it/s 
42 1624 1.008e+00 7.6it/s 
43 1662 9.961e-01 7.6it/s 
44 1700 9.883e-01 7.6it/s 
45 1739 9.531e-01 7.6it/s 
46 1777 1.023e+00 7.5it/s 
47 1815 9.727e-01 7.4it/s 
48 1853 9.844e-01 7.5it/s 
49 1891 9.531e-01 7.5it/s 
50 1929 9.570e-01 7.5it/s 
51 1967 9.219e-01 7.6it/s 
52 2005 8.711e-01 7.6it/s 
53 2043 8.867e-01 7.4it/s 
54 2081 9.062e-01 7.5it/s 
55 2119 8.594e-01 7.5it/s 
56 2157 8.242e-01 7.5it/s 
57 2194 8.477e-01 7.4it/s 
58 2231 7.734e-01 7.3it/s 
59 2269 8.594e-01 7.4it/s 
60 2306 8.125e-01 7.4it/s 
61 2344 7.617e-01 7.4it/s 
62 2381 8.242e-01 7.4it/s 
63 2418 7.539e-01 7.3it/s 
64 2455 7.930e-01 7.3it/s 
65 2492 7.930e-01 7.4it/s 
66 2529 7.969e-01 7.3it/s 
67 2566 7.617e-01 7.4it/s 
68 2604 6.875e-01 7.4it/s 
69 2641 7.500e-01 7.4it/s 
70 2678 7.461e-01 7.4it/s 
71 2715 6.953e-01 7.4it/s 
72 2753 6.758e-01 7.4it/s 
73 2791 6.836e-01 7.4it/s 
74 2828 6.914e-01 7.4it/s 
75 2866 6.211e-01 7.4it/s 
76 2904 6.641e-01 7.4it/s 
77 2942 6.367e-01 7.4it/s 
78 2979 6.367e-01 7.3it/s 
79 3016 6.172e-01 7.4it/s 
80 3054 6.094e-01 7.4it/s 
81 3092 6.133e-01 7.4it/s 
82 3129 5.742e-01 7.4it/s 
83 3166 5.938e-01 7.4it/s 
84 3204 6.094e-01 7.4it/s 
85 3241 5.664e-01 7.4it/s 
86 3278 5.938e-01 7.4it/s 
87 3316 5.625e-01 7.4it/s 
88 3353 5.898e-01 7.4it/s 
89 3390 5.273e-01 7.3it/s 
90 3428 5.352e-01 7.5it/s 
91 3466 5.352e-01 7.4it/s 
92 3504 5.312e-01 7.4it/s 
93 3541 4.941e-01 7.3it/s 
94 3578 4.805e-01 7.3it/s 
95 3615 4.805e-01 7.4it/s 
96 3652 5.000e-01 7.3it/s 
97 3689 4.785e-01 7.4it/s 
98 3727 4.980e-01 7.4it/s 
99 3764 4.434e-01 7.3it/s 
100 3801 4.492e-01 7.4it/s 
101 3839 4.082e-01 7.4it/s 
102 3877 4.453e-01 7.4it/s 
103 3914 4.531e-01 7.3it/s 
104 3952 4.336e-01 7.4it/s 
105 3990 4.004e-01 7.4it/s 
xx 3999 4.258e-01  (done)
config: tanh
00 0000 4.188e+00 0.0it/s 
01 0039 3.062e+00 7.6it/s 
02 0077 2.516e+00 7.5it/s 
03 0114 2.438e+00 7.4it/s 
04 0151 2.203e+00 7.3it/s 
05 0188 2.047e+00 7.4it/s 
06 0225 1.906e+00 7.4it/s 
07 0264 1.766e+00 7.6it/s 
08 0302 1.664e+00 7.4it/s 
09 0340 1.594e+00 7.5it/s 
10 0378 1.516e+00 7.5it/s 
11 0416 1.477e+00 7.6it/s 
12 0454 1.445e+00 7.5it/s 
13 0492 1.383e+00 7.5it/s 
14 0530 1.375e+00 7.6it/s 
15 0568 1.352e+00 7.5it/s 
16 0606 1.305e+00 7.4it/s 
17 0644 1.297e+00 7.5it/s 
18 0682 1.281e+00 7.5it/s 
19 0720 1.211e+00 7.5it/s 
20 0758 1.203e+00 7.5it/s 
21 0796 1.172e+00 7.6it/s 
22 0834 1.172e+00 7.6it/s 
23 0872 1.148e+00 7.5it/s 
24 0910 1.125e+00 7.5it/s 
25 0948 1.148e+00 7.6it/s 
26 0986 1.086e+00 7.5it/s 
27 1024 1.070e+00 7.5it/s 
28 1062 1.109e+00 7.4it/s 
29 1099 1.086e+00 7.4it/s 
30 1137 1.062e+00 7.4it/s 
31 1175 9.844e-01 7.4it/s 
32 1212 9.844e-01 7.4it/s 
33 1250 9.844e-01 7.5it/s 
34 1288 9.531e-01 7.4it/s 
35 1325 1.000e+00 7.4it/s 
36 1363 9.336e-01 7.4it/s 
37 1401 9.453e-01 7.4it/s 
38 1438 9.531e-01 7.4it/s 
39 1475 9.531e-01 7.4it/s 
40 1513 8.945e-01 7.4it/s 
41 1550 9.219e-01 7.4it/s 
42 1588 8.359e-01 7.4it/s 
43 1626 8.750e-01 7.4it/s 
44 1664 8.320e-01 7.4it/s 
45 1702 8.711e-01 7.5it/s 
46 1740 8.398e-01 7.5it/s 
47 1778 7.930e-01 7.5it/s 
48 1816 7.656e-01 7.4it/s 
49 1854 7.812e-01 7.5it/s 
50 1892 7.148e-01 7.5it/s 
51 1929 8.047e-01 7.4it/s 
52 1967 7.344e-01 7.5it/s 
53 2005 7.148e-01 7.6it/s 
54 2043 7.266e-01 7.5it/s 
55 2081 7.188e-01 7.4it/s 
56 2119 6.680e-01 7.4it/s 
57 2157 6.367e-01 7.5it/s 
58 2195 6.094e-01 7.4it/s 
59 2233 6.914e-01 7.5it/s 
60 2271 6.289e-01 7.5it/s 
61 2309 6.562e-01 7.5it/s 
62 2347 6.055e-01 7.4it/s 
63 2385 6.055e-01 7.5it/s 
64 2423 5.273e-01 7.5it/s 
65 2461 5.977e-01 7.5it/s 
66 2499 6.367e-01 7.5it/s 
67 2537 5.781e-01 7.5it/s 
68 2575 5.312e-01 7.6it/s 
69 2613 5.781e-01 7.5it/s 
70 2651 5.312e-01 7.5it/s 
71 2689 5.586e-01 7.6it/s 
72 2727 5.039e-01 7.6it/s 
73 2765 5.156e-01 7.6it/s 
74 2803 4.746e-01 7.6it/s 
75 2841 4.824e-01 7.5it/s 
76 2879 5.000e-01 7.5it/s 
77 2917 4.941e-01 7.5it/s 
78 2955 4.512e-01 7.5it/s 
79 2993 4.590e-01 7.5it/s 
80 3031 4.688e-01 7.5it/s 
81 3069 4.258e-01 7.5it/s 
82 3107 4.141e-01 7.5it/s 
83 3145 4.316e-01 7.5it/s 
84 3183 4.258e-01 7.5it/s 
85 3221 4.160e-01 7.5it/s 
86 3259 4.082e-01 7.5it/s 
87 3297 4.023e-01 7.5it/s 
88 3335 4.219e-01 7.5it/s 
89 3373 3.984e-01 7.5it/s 
90 3411 3.984e-01 7.5it/s 
91 3449 3.613e-01 7.6it/s 
92 3487 3.809e-01 7.6it/s 
93 3526 3.438e-01 7.6it/s 
94 3564 3.555e-01 7.6it/s 
95 3602 3.438e-01 7.6it/s 
96 3641 3.594e-01 7.6it/s 
97 3680 3.574e-01 7.6it/s 
98 3718 3.340e-01 7.6it/s 
99 3756 3.281e-01 7.6it/s 
100 3795 3.262e-01 7.6it/s 
101 3834 3.340e-01 7.6it/s 
102 3872 3.281e-01 7.6it/s 
103 3911 3.398e-01 7.6it/s 
104 3950 3.008e-01 7.6it/s 
105 3988 3.086e-01 7.6it/s 
xx 3999 3.086e-01  (done)
"""
