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
        def make_layer(init):
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
                       vocab=65, dropout_p=0.2, qk_scale=0.1, emb_scale=0.03)
    models = {f'v_scale: {s}': init_gpt(**base_params, v_scale=s) for s in utils.SCALES}
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
        model, opt_state = utils.optimize(model, opt_state, update, iter_count=250)
        outputs[name] = dict(model=model, opt_state=opt_state)
    return outputs


"""
config: v_scale: 0.003
00 0000 4.406e+00 0.0it/s 
01 0039 2.703e+00 7.8it/s 
02 0079 2.516e+00 7.8it/s 
03 0118 2.312e+00 7.8it/s 
04 0157 2.125e+00 7.7it/s 
05 0196 2.000e+00 7.7it/s 
06 0234 1.906e+00 7.5it/s 
xx 0249 1.914e+00  (done)
config: v_scale: 0.01
00 0000 4.406e+00 0.0it/s 
01 0040 2.703e+00 7.9it/s 
02 0080 2.516e+00 7.8it/s 
03 0120 2.344e+00 7.9it/s 
04 0160 2.109e+00 7.8it/s 
05 0199 2.016e+00 7.8it/s 
06 0238 1.891e+00 7.8it/s 
xx 0249 1.898e+00  (done)
config: v_scale: 0.03
00 0000 4.406e+00 0.0it/s 
01 0040 2.703e+00 7.8it/s 
02 0079 2.500e+00 7.7it/s 
03 0119 2.281e+00 7.8it/s 
04 0158 2.094e+00 7.8it/s 
05 0197 2.016e+00 7.7it/s 
06 0236 1.945e+00 7.6it/s 
xx 0249 1.906e+00  (done)
config: v_scale: 0.1
00 0000 4.406e+00 0.0it/s 
01 0040 2.750e+00 7.8it/s 
02 0079 2.516e+00 7.7it/s 
03 0118 2.469e+00 7.7it/s 
04 0157 2.234e+00 7.7it/s 
05 0196 2.047e+00 7.7it/s 
06 0235 1.969e+00 7.7it/s 
xx 0249 1.953e+00  (done)
config: v_scale: 0.3
00 0000 4.438e+00 0.0it/s 
01 0040 2.891e+00 7.8it/s 
02 0079 2.531e+00 7.8it/s 
03 0118 2.484e+00 7.8it/s 
04 0157 2.469e+00 7.8it/s 
05 0197 2.328e+00 7.8it/s 
06 0236 2.078e+00 7.8it/s 
xx 0249 2.047e+00  (done)
config: v_scale: 1.0
00 0000 4.562e+00 0.0it/s 
01 0039 2.969e+00 7.8it/s 
02 0078 2.562e+00 7.7it/s 
03 0118 2.484e+00 7.8it/s 
04 0157 2.453e+00 7.8it/s 
05 0196 2.344e+00 7.8it/s 
06 0236 2.141e+00 7.8it/s 
xx 0249 2.109e+00  (done)
config: v_scale: 3.0
00 0000 5.000e+00 0.0it/s 
01 0040 2.984e+00 7.9it/s 
02 0080 2.609e+00 7.8it/s 
03 0120 2.594e+00 7.9it/s 
04 0160 2.484e+00 7.9it/s 
05 0199 2.359e+00 7.8it/s 
06 0239 2.219e+00 7.8it/s 
xx 0249 2.203e+00  (done)
"""
