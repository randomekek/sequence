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
    def Attention(x, key, qk_proj, v_proj, out, heads: int, dropout_p: float):
        x_norm = jax.vmap(rms_norm)(x)
        parts = einsum(x_norm, qk_proj, 'L E, E HsplitD -> L HsplitD')
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

    def init_gpt(seq_length, layer_count, embed_size, heads, vocab, dropout_p, qkv_scale, emb_scale):
        init = funtree.Initializer(jr.PRNGKey(0))
        def make_layer(init):
            return Block(
                attn=Attention(
                    qk_proj=init.glorot_normal([embed_size, embed_size * 2]) * qkv_scale,
                    v_proj=init.glorot_normal([embed_size, embed_size]) * qkv_scale,
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
        model, opt_state = utils.optimize(model, opt_state, update, iter_count=2500)
        outputs[name] = dict(model=model, opt_state=opt_state)
    return outputs


"""
config: base
00 0000 4.406e+00 0.0it/s 
01 0043 2.688e+00 8.5it/s 
02 0086 2.516e+00 8.4it/s 
03 0129 2.375e+00 8.5it/s 
04 0172 2.125e+00 8.4it/s 
05 0215 2.016e+00 8.4it/s 
06 0257 1.953e+00 8.4it/s 
07 0296 1.836e+00 7.7it/s 
08 0335 1.758e+00 7.6it/s 
09 0374 1.656e+00 7.6it/s 
10 0412 1.625e+00 7.6it/s 
11 0451 1.578e+00 7.6it/s 
12 0490 1.531e+00 7.6it/s 
13 0528 1.453e+00 7.6it/s 
14 0566 1.445e+00 7.6it/s 
15 0605 1.414e+00 7.6it/s 
16 0644 1.414e+00 7.6it/s 
17 0682 1.438e+00 7.6it/s 
18 0720 1.328e+00 7.6it/s 
19 0759 1.344e+00 7.6it/s 
20 0797 1.359e+00 7.6it/s 
21 0835 1.344e+00 7.5it/s 
22 0873 1.250e+00 7.6it/s 
23 0912 1.297e+00 7.6it/s 
24 0950 1.250e+00 7.6it/s 
25 0989 1.281e+00 7.6it/s 
26 1028 1.234e+00 7.6it/s 
27 1066 1.234e+00 7.6it/s 
28 1104 1.250e+00 7.5it/s 
29 1142 1.180e+00 7.6it/s 
30 1181 1.164e+00 7.6it/s 
31 1219 1.164e+00 7.5it/s 
32 1257 1.109e+00 7.5it/s 
33 1295 1.156e+00 7.5it/s 
34 1333 1.125e+00 7.5it/s 
35 1371 1.141e+00 7.5it/s 
36 1409 1.141e+00 7.5it/s 
37 1447 1.086e+00 7.5it/s 
38 1485 1.047e+00 7.5it/s 
39 1523 1.055e+00 7.5it/s 
40 1561 1.031e+00 7.5it/s 
41 1599 1.039e+00 7.5it/s 
42 1637 1.008e+00 7.5it/s 
43 1675 1.023e+00 7.5it/s 
44 1713 1.000e+00 7.5it/s 
45 1751 9.883e-01 7.5it/s 
46 1789 9.688e-01 7.5it/s 
47 1827 9.805e-01 7.5it/s 
48 1865 9.531e-01 7.5it/s 
49 1903 9.453e-01 7.6it/s 
50 1941 9.688e-01 7.6it/s 
51 1979 9.414e-01 7.6it/s 
52 2017 8.711e-01 7.6it/s 
53 2055 8.867e-01 7.6it/s 
54 2093 8.711e-01 7.6it/s 
55 2132 8.828e-01 7.6it/s 
56 2170 8.281e-01 7.6it/s 
57 2208 8.555e-01 7.5it/s 
58 2246 8.438e-01 7.6it/s 
59 2284 8.281e-01 7.6it/s 
60 2322 8.008e-01 7.5it/s 
61 2360 8.125e-01 7.6it/s 
62 2399 7.578e-01 7.6it/s 
63 2437 8.633e-01 7.6it/s 
64 2475 7.891e-01 7.6it/s 
xx 2499 7.695e-01  (done)
"""
