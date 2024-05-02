# %%
import dataclasses
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import utils


# %%
def main():
    ignore_value = -1

    def recall_task(key, length, fixed_query_vocab, ctx_query_vocab, value_vocab, batches):
        # vocab = [fixed_query_vocab, ctx_query_vocab, value_vocab]
        fixed_key, batch_key = jr.split(key)
        value_vocab_min = fixed_query_vocab + ctx_query_vocab
        value_vocab_max = value_vocab_min + value_vocab
        fixed_map = jr.randint(fixed_key, (fixed_query_vocab,), value_vocab_min, value_vocab_max)

        def make_batch(key):
            ctx_key, key_key = jr.split(key)
            ctx_map = jr.randint(ctx_key, (ctx_query_vocab,), value_vocab_min, value_vocab_max)
            total_map = jnp.concat([fixed_map, ctx_map])
            keys = jr.randint(key_key, (length,), 0, fixed_query_vocab + ctx_query_vocab)
            values = jnp.take(total_map, keys)
            input = jnp.ravel(jnp.stack([keys, values]), order='F')
            mask = jnp.arange(input.shape[0]) % 2 == 1
            output = jnp.where(mask, ignore_value, jnp.concatenate([input[1:], jnp.array([ignore_value])]))
            return (input, output)
        return jax.vmap(make_batch)(jr.split(batch_key, batches))

    class Model(eqx.Module):
        pass

    def accuracy_fn(batch):  # ([B, L], [B, L])
        x, y = batch
        @eqx.filter_jit
        def fn(model):
            prediction = jnp.argmax(jax.vmap(model)(x))  # argmax(softmax(x)) == argmax(x)
            ignore_mask = y != ignore_value
            return jnp.mean(ignore_mask * (prediction == y), axis=(0, 1))
        return fn

    @eqx.filter_value_and_grad
    def loss_fn(model, x, y):  # model, model, [B, L], [B, L]
        logits = jax.vmap(model)(x)  # [B, L, C]
        prediction = jnp.clip(jax.nn.softmax(logits, axis=2), 1e-7, 1.0 - 1e-7)
        ignore_mask = y != ignore_value
        return -jnp.mean(ignore_mask * y * jnp.log(prediction))

    @eqx.filter_jit
    def update(b, model, opt_state):
        x, y = recall_task(key=jr.PRNGKey(b), length=10, fixed_query_vocab=5,
                           ctx_query_vocab=5, value_vocab=10, batches=2)
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    model = Model(jr.PRNGKey(0))
    optimizer = optax.adam(1e-4)  # needs to be lower for more complex models
    opt_state = optimizer.init(model)
    accuracy_set = recall_task(key=jr.PRNGKey(-1), length=10, fixed_query_vocab=5,
                               ctx_query_vocab=5, value_vocab=10, batches=2)

    return utils.optimize(model, opt_state, update, accuracy_fn(accuracy_set))


final_model, final_opt_state = main()
