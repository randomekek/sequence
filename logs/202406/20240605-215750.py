"""
test if we can replace gelu with a x-1.5tanh(x)
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
        activation = lambda x: x - 2.5 * jax.nn.tanh(x)
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

    def init_gpt(seq_length, layer_count, embed_size, heads, vocab, dropout_p, qk_scale, emb_scale, v_scale):
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
                    up=init.he_normal([embed_size, 4 * embed_size]),
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
                       vocab=65, dropout_p=0.2, qk_scale=0.1, emb_scale=0.03, v_scale=0.01)
    models = {'base': init_gpt(**base_params)}
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
        model, opt_state, interrupted = utils.optimize(model, opt_state, update, iter_count=2500)
        outputs[name] = dict(model=model, opt_state=opt_state)
        if interrupted:
            break
    return outputs


"""
config: base
00 0000 4.750e+00 0.0it/s 
01 0039 3.062e+00 7.7it/s 
02 0078 2.656e+00 7.7it/s 
03 0117 2.578e+00 7.8it/s 
04 0157 2.531e+00 7.9it/s 
05 0197 2.516e+00 7.9it/s 
06 0237 2.500e+00 7.9it/s 
07 0277 2.484e+00 7.9it/s 
08 0317 2.469e+00 7.8it/s 
09 0356 2.312e+00 7.8it/s 
10 0396 2.156e+00 7.9it/s 
11 0436 2.047e+00 7.8it/s 
12 0476 1.961e+00 7.9it/s 
13 0516 1.953e+00 7.8it/s 
14 0556 1.852e+00 7.9it/s 
15 0596 1.859e+00 7.9it/s 
16 0636 1.828e+00 7.8it/s 
17 0676 1.805e+00 7.9it/s 
18 0716 1.766e+00 7.9it/s 
19 0756 1.750e+00 7.9it/s 
20 0796 1.711e+00 7.8it/s 
21 0835 1.680e+00 7.7it/s 
22 0874 1.617e+00 7.7it/s 
23 0913 1.609e+00 7.6it/s 
24 0952 1.602e+00 7.6it/s 
25 0991 1.570e+00 7.7it/s 
26 1031 1.586e+00 7.8it/s 
27 1071 1.484e+00 7.8it/s 
28 1111 1.516e+00 7.9it/s 
29 1151 1.484e+00 7.9it/s 
30 1191 1.484e+00 7.9it/s 
31 1231 1.469e+00 7.8it/s 
32 1270 1.438e+00 7.8it/s 
33 1309 1.453e+00 7.8it/s 
34 1348 1.422e+00 7.8it/s 
35 1387 1.383e+00 7.8it/s 
36 1426 1.445e+00 7.8it/s 
37 1465 1.352e+00 7.8it/s 
38 1504 1.352e+00 7.8it/s 
39 1543 1.367e+00 7.8it/s 
40 1582 1.367e+00 7.8it/s 
41 1621 1.352e+00 7.8it/s 
42 1660 1.344e+00 7.8it/s 
43 1699 1.312e+00 7.8it/s 
44 1738 1.359e+00 7.8it/s 
45 1777 1.281e+00 7.8it/s 
46 1817 1.289e+00 7.8it/s 
47 1857 1.289e+00 7.8it/s 
48 1897 1.234e+00 7.9it/s 
49 1937 1.211e+00 7.8it/s 
50 1977 1.234e+00 7.9it/s 
51 2017 1.219e+00 7.9it/s 
52 2057 1.227e+00 7.8it/s 
53 2097 1.211e+00 7.8it/s 
54 2137 1.227e+00 7.9it/s 
55 2177 1.188e+00 7.9it/s 
56 2217 1.172e+00 7.8it/s 
57 2257 1.211e+00 7.9it/s 
58 2297 1.172e+00 7.9it/s 
59 2337 1.164e+00 7.8it/s 
60 2377 1.156e+00 7.8it/s 
61 2417 1.102e+00 7.9it/s 
62 2457 1.094e+00 7.9it/s 
63 2497 1.156e+00 7.9it/s 
xx 2499 1.094e+00  (done)
"""
