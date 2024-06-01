# %%
from einops import einsum, rearrange, reduce, repeat
from funtree import dropout, norm, rms_norm
import funtree
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import optax
import utils


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
        # Disable DLN:
        # return jnp.zeros([x.shape[0], update_proj.shape[0]], dtype=jnp.bfloat16)
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

    @funtree.makefun
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

    @funtree.makefun
    def Block(x, key, attn, mlp, dln):
        dln_key, attn_key, mlp_key = jr.split(key, 3)
        h = dln(x, key=dln_key)
        x = x + attn(jnp.concat([x, h], axis=-1), key=attn_key)
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
        'base': init_gpt(**base_params, dln_size=1000, history_len=1),
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
        model, opt_state = utils.optimize(model, opt_state, update, iter_count=250)
        outputs[name] = dict(model=model, opt_state=opt_state)
    return outputs


outputs = utils.run(main, 'set the history down to 1')

# %%

meta = json.load(open('shakespeare_char/meta.json'))
char_map = jnp.array([ord(c) for c in meta['chars']])
model = outputs['linear']['model']


def as_text(vals):
    return ''.join(chr(c) for c in char_map[vals]).replace('\n', '¶')


def predict():
    for a in range(25):
        k = jax.random.PRNGKey(a)
        task = tasks(k)[0][0]
        print(as_text(task))
        print(' ' + as_text(jnp.argmax(model(task, k), axis=-1)))


def generate():
    x = jnp.array([15])
    k = jax.random.PRNGKey(0)
    for i in range(70):
        next = jnp.argmax(model(x, k)[-1:, :], axis=-1)
        x = jnp.concatenate([x, next])
    print(as_text(x))
