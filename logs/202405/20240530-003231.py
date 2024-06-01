"""
compare with midGPT
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
    def MLP(x, key, up, down, dropout_p: float):
        x_norm = jax.vmap(rms_norm)(x)
        expanded = jax.nn.gelu(einsum(x_norm, up, 'L E, E U -> L U'))
        lowered = einsum(expanded, down, 'L U, U E -> L E')
        return dropout(lowered, key, dropout_p)

    @funtree.makefun
    def Attention(x, key, qkv, out, heads: int, dropout_p: float):
        x_norm = jax.vmap(rms_norm)(x)
        parts = einsum(x_norm, qkv, 'L E, E HsplitD -> L HsplitD')
        k, q, v = rearrange(parts, 'L (split H D) -> split H L D', split=3, H=heads)
        q, k = jax.vmap(jax.vmap(norm))(q), jax.vmap(jax.vmap(norm))(k)
        H, L, D = k.shape
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

    def init_gpt(seq_length, layer_count, embed_size, heads, vocab, dropout_p, qkv_scale, emb_scale):
        init = funtree.Initializer(jr.PRNGKey(0))
        def make_layer(init):
            return Block(
                attn=Attention(
                    qkv=init.glorot_normal([embed_size, embed_size * 3]) * qkv_scale,
                    out=init.glorot_normal([embed_size, embed_size]),
                    heads=heads,
                    dropout_p=dropout_p),
                mlp=MLP(
                    up=init.glorot_normal([embed_size, 4 * embed_size]),
                    down=init.glorot_normal([4 * embed_size, embed_size]),
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
                       vocab=65, dropout_p=0.2, qkv_scale=0.1, emb_scale=0.03)
    models = {
        'base': init_gpt(**base_params),
    }
    outputs = {}
    for name, model in models.items():
        print(f'config: {name}')
        decay_weights = funtree.funmap(model, {
            Block: lambda **kw: kw,
            Attention: lambda **kw: kw | dict(qkv=True, out=True),
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
        model, opt_state = utils.optimize(model, opt_state, update, iter_count=2500)
        outputs[name] = dict(model=model, opt_state=opt_state)
    return outputs


"""
config: base
00 0000 4.344e+00 0.0it/s 
01 0040 2.781e+00 8.0it/s 
02 0080 2.516e+00 8.0it/s 
03 0121 2.375e+00 8.0it/s 
04 0161 2.156e+00 8.0it/s 
05 0201 2.031e+00 8.0it/s 
06 0241 1.945e+00 8.0it/s 
07 0281 1.883e+00 8.0it/s 
08 0321 1.797e+00 7.9it/s 
09 0361 1.750e+00 8.0it/s 
10 0401 1.672e+00 8.0it/s 
11 0441 1.594e+00 8.0it/s 
12 0481 1.547e+00 7.9it/s 
13 0521 1.531e+00 8.0it/s 
14 0561 1.477e+00 8.0it/s 
15 0601 1.445e+00 7.9it/s 
16 0641 1.469e+00 8.0it/s 
17 0681 1.430e+00 7.9it/s 
18 0721 1.406e+00 7.9it/s 
19 0761 1.375e+00 7.9it/s 
20 0801 1.352e+00 8.0it/s 
21 0841 1.328e+00 8.0it/s 
22 0881 1.273e+00 7.9it/s 
23 0921 1.281e+00 7.9it/s 
24 0961 1.305e+00 7.9it/s 
25 1001 1.281e+00 8.0it/s 
26 1041 1.250e+00 7.9it/s 
27 1081 1.227e+00 7.9it/s 
28 1121 1.234e+00 8.0it/s 
29 1161 1.227e+00 7.9it/s 
30 1201 1.242e+00 7.9it/s 
31 1241 1.234e+00 7.9it/s 
32 1281 1.195e+00 7.9it/s 
33 1321 1.164e+00 7.8it/s 
34 1361 1.141e+00 7.8it/s 
35 1401 1.102e+00 7.8it/s 
36 1441 1.078e+00 7.8it/s 
37 1481 1.094e+00 7.8it/s 
38 1521 1.070e+00 7.9it/s 
39 1561 1.047e+00 7.8it/s 
40 1601 1.055e+00 7.8it/s 
41 1641 1.047e+00 7.8it/s 
42 1681 1.070e+00 7.9it/s 
43 1721 1.047e+00 7.8it/s 
44 1761 9.961e-01 7.8it/s 
45 1801 1.000e+00 7.9it/s 
46 1841 1.031e+00 7.8it/s 
47 1881 9.688e-01 7.9it/s 
48 1921 9.375e-01 7.9it/s 
49 1961 9.453e-01 7.9it/s 
50 2001 9.102e-01 8.0it/s 
51 2041 9.297e-01 7.9it/s 
52 2081 8.984e-01 7.9it/s 
53 2121 8.945e-01 8.0it/s 
54 2161 8.555e-01 7.9it/s 
55 2201 8.633e-01 7.8it/s 
56 2241 8.867e-01 7.9it/s 
57 2281 8.398e-01 8.0it/s 
58 2321 7.930e-01 7.9it/s 
59 2361 8.125e-01 7.9it/s 
60 2401 8.320e-01 8.0it/s 
61 2441 8.008e-01 7.9it/s 
62 2481 8.398e-01 7.9it/s 
xx 2499 7.852e-01  (done)
"""
