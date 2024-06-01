"""
compare with midGPT
"""

def main():
    from einops import einsum, rearrange, reduce, repeat
    from funtree import dropout, norm, rms_norm
    import funtree
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import json
    import optax
    import utils

    @funtree.makefun
    def MLP(x, key, up, down, dropout_p: float):
        x_norm = jax.vmap(rms_norm)(x)
        expanded = jax.nn.gelu(einsum(x_norm, up, 'L E, E U -> L U'))
        lowered = einsum(expanded, down, 'L U, U E -> L E')
        return dropout(lowered, key, dropout_p)

    @funtree.makefun
    def DLN(x, key, update_proj, input_proj, dropout_p: float):
        # update_proj: P, input_proj: E P, returns L P
        x_norm = jax.vmap(rms_norm)(x)
        vals = einsum(x_norm, input_proj, 'L E, E P -> L P')
        L, P = vals.shape
        updates = repeat(update_proj, 'P -> L P', L=L)
        pairs = jnp.stack([updates, vals])
        def inner(a, b):
            pa, va = a
            pb, vb = b
            return jnp.stack([pa * pb, va * pb + vb])
        result = jax.lax.associative_scan(inner, pairs, axis=1)  # L 2 P
        result = result[0, :, :]
        result = dropout(result, key, dropout_p)
        return result

    @ funtree.makefun
    def Attention(x, key, qk_proj, v_proj, out, history_len: int, heads: int, dropout_p: float):
        x_norm = jax.vmap(rms_norm)(x)
        parts = einsum(x_norm, qk_proj, 'L E, E splitHD -> L splitHD')
        k, q = rearrange(parts, 'L (split H D) -> split H L D', split=2, H=heads)
        q, k = jax.vmap(jax.vmap(norm))(q), jax.vmap(jax.vmap(norm))(k)
        H, L, D = k.shape
        v = rearrange(einsum(x_norm, v_proj, 'L E, E HD -> L HD'), 'L (H D) -> H L D', H=heads)
        mask = jnp.tril(jnp.ones([L, L])) - jnp.tril(jnp.ones([L, L]), -history_len)
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

    @ funtree.makefun
    def Block(x, key, attn, mlp, dln):
        dln_key, attn_key, mlp_key = jr.split(key, 3)
        h = dln(x, key=dln_key)
        x = x + attn(jnp.concat([x, h], axis=-1), key=attn_key)
        x = x + mlp(x, key=mlp_key)
        return x

    @ funtree.makefun
    def GPT(x, key, embedding, positional, layers, unembed):
        L = x.shape[0]
        hidden = embedding[x] + positional[:L, :]
        for layer, k in funtree.zipkey(layers, key):
            hidden = hidden + layer(hidden, key=k)
        logits = einsum(unembed, hidden, 'E O, L E -> L O')
        return logits

    def init_gpt(seq_length, layer_count, embed_size, dln_size, history_len, heads, vocab, dropout_p, qk_scale, emb_scale, v_scale):
        init = funtree.Initializer(jr.PRNGKey(0))
        def make_layer(init: funtree.Initializer):
            return Block(
                attn=Attention(
                    qk_proj=init.glorot_normal([embed_size + dln_size, embed_size * 2]) * qk_scale,
                    v_proj=init.glorot_normal([embed_size + dln_size, embed_size]) * v_scale,
                    out=init.glorot_normal([embed_size, embed_size]),
                    history_len=history_len,
                    heads=heads,
                    dropout_p=dropout_p),
                mlp=MLP(
                    up=init.glorot_normal([embed_size, 4 * embed_size]),
                    down=init.glorot_normal([4 * embed_size, embed_size]),
                    dropout_p=dropout_p),
                dln=DLN(
                    update_proj=1 - init.uniform([dln_size], 0.95, 0.999),
                    input_proj=init.glorot_normal([embed_size, dln_size]),
                    dropout_p=dropout_p))
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

    @ jax.jit
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
        'base': init_gpt(**base_params, dln_size=100, history_len=5),
    }
    outputs = {}
    for name, model in models.items():
        print(f'config: {name}')
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(0.9, 0.99),
            optax.add_decayed_weights(0.1),
            optax.scale_by_schedule(optax.warmup_cosine_decay_schedule(
                init_value=0,
                peak_value=1e-3,
                warmup_steps=100,
                decay_steps=5000,
                end_value=1e-4)),
            optax.scale(-1))
        opt_state = optimizer.init(model)
        model, opt_state = utils.optimize(model, opt_state, update, iter_count=2500)
        outputs[name] = dict(model=model, opt_state=opt_state)
    return outputs


"""
config: base
00 0000 4.406e+00 0.0it/s 
01 0035 2.844e+00 6.9it/s 
02 0070 2.406e+00 7.0it/s 
03 0105 2.125e+00 7.0it/s 
04 0140 1.930e+00 6.9it/s 
05 0175 1.789e+00 6.9it/s 
06 0210 1.680e+00 6.9it/s 
07 0245 1.602e+00 6.9it/s 
08 0280 1.555e+00 6.9it/s 
09 0315 1.570e+00 6.9it/s 
10 0350 1.500e+00 6.9it/s 
11 0385 1.445e+00 6.9it/s 
12 0420 1.469e+00 6.9it/s 
13 0455 1.445e+00 6.9it/s 
14 0490 1.398e+00 6.9it/s 
15 0525 1.375e+00 6.8it/s 
16 0560 1.391e+00 6.8it/s 
17 0595 1.336e+00 6.8it/s 
18 0630 1.359e+00 6.8it/s 
19 0665 1.320e+00 6.8it/s 
20 0700 1.289e+00 6.8it/s 
21 0735 1.266e+00 6.8it/s 
22 0770 1.234e+00 6.8it/s 
23 0805 1.266e+00 6.8it/s 
24 0840 1.281e+00 6.8it/s 
25 0875 1.266e+00 6.8it/s 
26 0910 1.250e+00 6.8it/s 
27 0945 1.203e+00 6.8it/s 
28 0980 1.227e+00 6.8it/s 
29 1015 1.211e+00 6.8it/s 
30 1050 1.227e+00 6.8it/s 
31 1084 1.203e+00 6.8it/s 
32 1119 1.195e+00 6.9it/s 
33 1154 1.180e+00 6.8it/s 
34 1189 1.195e+00 6.9it/s 
35 1224 1.195e+00 6.9it/s 
36 1259 1.172e+00 6.8it/s 
37 1294 1.133e+00 6.8it/s 
38 1329 1.125e+00 6.8it/s 
39 1364 1.117e+00 6.9it/s 
40 1399 1.125e+00 6.8it/s 
41 1434 1.102e+00 6.8it/s 
42 1468 1.086e+00 6.8it/s 
43 1502 1.094e+00 6.7it/s 
44 1536 1.094e+00 6.8it/s 
45 1571 1.117e+00 6.8it/s 
46 1605 1.109e+00 6.7it/s 
47 1639 1.070e+00 6.6it/s 
48 1673 1.070e+00 6.7it/s 
49 1707 1.062e+00 6.7it/s 
50 1741 1.070e+00 6.8it/s 
51 1775 1.070e+00 6.7it/s 
52 1809 1.016e+00 6.7it/s 
53 1843 1.008e+00 6.7it/s 
54 1877 1.078e+00 6.6it/s 
55 1911 1.000e+00 6.8it/s 
56 1946 1.055e+00 6.8it/s 
57 1981 9.922e-01 6.9it/s 
58 2016 1.016e+00 6.8it/s 
59 2051 9.883e-01 6.9it/s 
60 2086 9.531e-01 6.9it/s 
61 2120 9.727e-01 6.8it/s 
62 2154 9.844e-01 6.8it/s 
63 2188 9.531e-01 6.8it/s 
64 2222 9.805e-01 6.8it/s 
65 2256 9.258e-01 6.8it/s 
66 2290 9.414e-01 6.8it/s 
67 2324 8.984e-01 6.8it/s 
68 2358 8.984e-01 6.8it/s 
69 2392 9.414e-01 6.8it/s 
70 2426 8.984e-01 6.8it/s 
71 2460 8.359e-01 6.8it/s 
72 2494 8.750e-01 6.8it/s 
xx 2499 8.867e-01  (done)
"""
