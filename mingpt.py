# %%
from einops import einsum, rearrange
from funtree import funmap, funtree, dropout, norm, rms_norm, Initializer, Update
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import optax
import utils


def main():

    @funtree
    def Mlp(x, key, up, down, dropout_p: float):
        x_norm = rms_norm(x)
        expanded = jax.nn.gelu(einsum(x_norm, up, 'L E, E U -> L U'))
        lowered = einsum(expanded, down, 'L U, U E -> L E')
        return dropout(lowered, key, dropout_p)

    @funtree
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

    @funtree
    def GPT(x, key, embedding, positional, layers, unembed):
        L = x.shape[0]
        hidden = embedding[x] + positional[:L, :]
        for layer, k in zipkey(layers, key):
            hidden = hidden + layer(hidden, key=k)
        logits = einsum(unembed, hidden, 'E O, L E -> L O')
        return logits

    def zipkey(items, key):
        return zip(items, jax.random.split(key, len(items)))

    def init_gpt_model(vocab, embedding, heads, layer_count, up, max_length):
        init = Initializer(jax.random.PRNGKey(0))
        def make_attention(init):
            return Attention(
                qkv=init.glorot_normal([embedding, 3*heads*embedding]),
                out=init.glorot_normal([heads*embedding, embedding]),
                heads=heads,
                dropout_p=0.1,
            )
        def make_mlp(init):
            return Mlp(
                up=init.glorot_normal([embedding, up*embedding]),
                down=init.glorot_normal([up*embedding, embedding]),
                dropout_p=0.1,
            )
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

    @jax.jit
    def update_with_task(x, y, model, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return loss, model, opt_state

    batch_size = 64
    seq_length = 128
    model = init_gpt_model(vocab=39, embedding=256, heads=6, layer_count=6, up=4, max_length=seq_length)
    decay_mask = funmap(model, {
        Attention: lambda **kw: Update(kw, qkv=True, out=True, key=False),
        Mlp: lambda **kw: Update(kw, up=True, down=True, key=False),
        GPT: lambda **kw: Update(kw, embedding=False, positional=False, unembed=False, key=False),
    })
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0, peak_value=3e-4, warmup_steps=40, decay_steps=400, end_value=1e-4)
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
    return final_model, final_opt_state


final_model, final_opt_state = main()
