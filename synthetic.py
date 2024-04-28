# %%
import jax
import jax.numpy as jnp
import jax.random as jr

def recall_task(
        key,
        length = 10,
        fixed_query_vocab = 5,
        ctx_query_vocab = 5,
        value_vocab = 10,
        batches = 10):
    # vocab = [fixed_query_vocab, ctx_query_vocab, value_vocab]
    fixed_key, batch_key = jr.split(key)
    value_vocab_min = fixed_query_vocab + ctx_query_vocab
    value_vocab_max = value_vocab_min + value_vocab
    fixed_map = jr.randint(fixed_key, (fixed_query_vocab,), value_vocab_min, value_vocab_max)
    def make_batch(key):
        ctx_key, key_key = jr.split(key)
        ctx_map = jr.randint(ctx_key, (ctx_query_vocab,), 0, value_vocab_min, value_vocab_max)
        total_map = jnp.concat([fixed_map, ctx_map])
        keys = jr.randint(key_key, (length,), 0, fixed_query_vocab + ctx_query_vocab)
        values = jnp.take(total_map, keys)
        return jnp.ravel(jnp.stack((keys, values)), order='F')
    return jax.vmap(make_batch)(jr.split(batch_key, batches))

print(recall_task(jr.PRNGKey(1)))
