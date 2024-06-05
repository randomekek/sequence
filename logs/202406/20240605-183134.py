"""
test if we can replace gelu with relu
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
        expanded = jax.nn.relu(einsum(x_norm, up, 'L E, E U -> L U'))
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
        try:
            model, opt_state = utils.optimize(model, opt_state, update, iter_count=2500)
            outputs[name] = dict(model=model, opt_state=opt_state)
        except KeyboardInterrupt:
            print('interrupt')
            return outputs
    return outputs


"""
config: base
00 0000 4.500e+00 0.0it/s 
01 0043 2.672e+00 8.6it/s 
02 0086 2.516e+00 8.5it/s 
03 0129 2.297e+00 8.6it/s 
04 0172 2.094e+00 8.5it/s 
05 0215 2.000e+00 8.6it/s 
06 0258 1.953e+00 8.5it/s 
07 0301 1.812e+00 8.5it/s 
08 0344 1.766e+00 8.5it/s 
09 0387 1.688e+00 8.5it/s 
10 0430 1.594e+00 8.5it/s 
11 0473 1.578e+00 8.5it/s 
12 0516 1.523e+00 8.5it/s 
13 0559 1.523e+00 8.5it/s 
14 0602 1.445e+00 8.5it/s 
15 0645 1.453e+00 8.5it/s 
16 0688 1.398e+00 8.5it/s 
17 0731 1.375e+00 8.5it/s 
18 0774 1.367e+00 8.4it/s 
19 0817 1.344e+00 8.4it/s 
20 0860 1.312e+00 8.4it/s 
21 0903 1.281e+00 8.4it/s 
22 0946 1.336e+00 8.5it/s 
23 0989 1.297e+00 8.5it/s 
24 1032 1.242e+00 8.5it/s 
25 1075 1.258e+00 8.5it/s 
26 1118 1.211e+00 8.5it/s 
27 1161 1.234e+00 8.5it/s 
28 1204 1.188e+00 8.4it/s 
29 1247 1.195e+00 8.5it/s 
30 1290 1.164e+00 8.5it/s 
31 1333 1.141e+00 8.5it/s 
32 1375 1.188e+00 8.4it/s 
33 1418 1.117e+00 8.4it/s 
34 1460 1.133e+00 8.4it/s 
35 1503 1.117e+00 8.4it/s 
36 1546 1.133e+00 8.4it/s 
37 1588 1.117e+00 8.4it/s 
38 1630 1.109e+00 8.4it/s 
39 1672 1.047e+00 8.4it/s 
40 1714 1.055e+00 8.4it/s 
41 1756 1.031e+00 8.4it/s 
42 1798 9.844e-01 8.4it/s 
43 1840 1.008e+00 8.4it/s 
44 1882 1.031e+00 8.4it/s 
45 1924 9.883e-01 8.4it/s 
46 1966 1.016e+00 8.4it/s 
47 2009 9.258e-01 8.4it/s 
48 2052 9.180e-01 8.4it/s 
49 2095 9.414e-01 8.5it/s 
50 2137 9.453e-01 8.4it/s 
51 2179 8.984e-01 8.3it/s 
interrupt
xx 2188 8.867e-01  (done)
"""
