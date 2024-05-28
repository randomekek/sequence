# %%
from dataclasses import dataclass
from einops import einsum, rearrange, reduce
from funtree import dropout, norm, rms_norm
import equinox as eqx
import funtree
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.random as jrandom
import json
import math
import optax
import typing as tp
import utils


def print_dtypes(vars):
    print('\n'.join(f'{k}: {v.dtype}{v.shape}' for k, v in vars.items() if hasattr(v, 'dtype')))
    raise Exception()


jnp = jax.numpy
KeyArray = tp.Any
Array = jax.numpy.ndarray
jrandom = jax.random

jnp, jrandom, vmap, jtu = jax.numpy, jax.random, jax.vmap, jax.tree_util
Array = jax.Array
KeyArray = tp.Any
P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding
Mesh = jax.sharding.Mesh
with_sharding_constraint = jax.lax.with_sharding_constraint


# # %%


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

    class Embedding(eqx.Module):
        """For some reason, default Embedding impl is slow under vmap+JIT."""
        V: int = eqx.field(static=True)  # num embeddings
        D: int = eqx.field(static=True)  # embedding size
        weight_VxD: Array

        def __init__(
                self, num_embeddings: int, embedding_size: int, weight: tp.Optional[Array] = None,
                *, key: tp.Optional[KeyArray] = None
        ):
            super().__init__()
            self.V, self.D = num_embeddings, embedding_size
            if weight is not None:
                self.weight_VxD = weight
            elif key is not None:
                self.weight_VxD = jrandom.normal(key, (self.V, self.D))
            else:
                raise ValueError("need weight or key to be not None")

        @jax.named_scope("Embedding")
        def __call__(self, x_T, *, key=None):
            return jnp.take(self.weight_VxD, x_T, axis=0)

    class Linear(eqx.Module):
        """Linear with trunc normal init."""
        weight_MxN: Array

        def __init__(
            self, in_features: int, out_features: int, weight: tp.Optional[Array] = None,
            *, key: tp.Optional[KeyArray] = None
        ):
            super().__init__()
            if weight is not None:
                self.weight_MxN = weight
            elif key is not None:
                self.weight_MxN = (1 / math.sqrt(in_features)) * jrandom.truncated_normal(
                    key, lower=-2, upper=2, shape=(out_features, in_features))
            else:
                raise ValueError("need weight or key to be not None")

        @jax.named_scope("Linear")
        def __call__(self, x_N: Array, *, key: KeyArray = None) -> Array:
            x_M = self.weight_MxN @ x_N
            return x_M

    class RMSNorm(eqx.Module):
        weight_M: tp.Optional[Array]
        eps: float

        def __init__(self, dim: int, use_weight=False, eps=1e-6):
            super().__init__()
            self.eps, self.weight_M = eps, None
            if use_weight:
                self.weight_M = jnp.ones((dim,))

        @jax.named_scope("RMSNorm")
        def __call__(self, x_M: Array) -> Array:
            out_M = x_M * jax.lax.rsqrt(jnp.mean(jnp.square(x_M), keepdims=True) + self.eps)
            if self.weight_M is not None:
                out_M = out_M * self.weight_M
            return out_M

    class MLP(eqx.Module):
        c_fc: Linear
        c_proj: Linear
        dropout: eqx.nn.Dropout
        ln2: RMSNorm

        def __init__(self, n_embd, dropout, key):
            key1, key2 = jrandom.split(key)
            self.c_fc = Linear(n_embd, 4 * n_embd, key=key1)
            self.c_proj = Linear(4 * n_embd, n_embd, key=key2)
            self.dropout = eqx.nn.Dropout(dropout)
            self.ln2 = RMSNorm(n_embd)

        @jax.named_scope('mlp')
        def __call__(self, x_D, key=None):
            x_D = self.ln2(x_D)
            x_D = jax.nn.gelu(self.c_fc(x_D))
            x_D = self.dropout(self.c_proj(x_D), inference=False, key=key)
            return x_D

    # target:
    # 460 loss 1.6
    # 690 loss 1.5
    class CausalSelfAttention(eqx.Module):
        n_head: int
        n_embd: int
        c_attn: Linear
        c_proj: Linear
        attn_dropout: eqx.nn.Dropout
        resid_dropout: eqx.nn.Dropout
        q_ln: eqx.nn.LayerNorm
        k_ln: eqx.nn.LayerNorm
        ln1: RMSNorm

        def __init__(self, n_embd, n_head, dropout, key):
            key1, key2 = jrandom.split(key)
            assert n_embd % n_head == 0
            self.n_head, self.n_embd = n_head, n_embd
            self.c_attn = Linear(n_embd, 3 * n_embd, key=key1)
            self.c_proj = Linear(n_embd, n_embd, key=key2)
            self.attn_dropout = eqx.nn.Dropout(dropout)
            self.resid_dropout = eqx.nn.Dropout(dropout)
            self.q_ln = eqx.nn.LayerNorm(n_embd // n_head, eps=1e-6, use_weight=False, use_bias=False)
            self.k_ln = eqx.nn.LayerNorm(n_embd // n_head, eps=1e-6, use_weight=False, use_bias=False)
            self.ln1 = RMSNorm(n_embd)

        @jax.named_scope('causal_sa')
        def __call__(self, x_TxD, inference=False, key=None):
            # -- x_norm = vmap(rms_norm)(x)
            # -- parts = einsum(x_norm, qkv, 'L E, E HsplitD -> L HsplitD')
            # -- k, q, v = rearrange(parts, 'L (split H D) -> split H L D', split=3, H=heads)
            # -- q, k = vmap(vmap(norm))(q), vmap(vmap(norm))(k)
            # H, L, D = k.shape
            # mask = jnp.tril(jnp.ones([L, L]))
            # similarity = einsum(k, q, 'H L D, H L2 D -> H L L2')
            # masked_similarity = jnp.where(mask, similarity, -jnp.inf)
            # attention = jax.nn.softmax(masked_similarity.astype(jnp.float32) / jnp.sqrt(D), axis=-1)
            # attention = attention.astype(jnp.bfloat16)
            # key1, key2 = jr.split(key)
            # attention = dropout(attention, key1, 0.2)
            # fetch = einsum(attention, v, 'H L L2, H L2 V -> H L V')
            # gather = rearrange(fetch, 'H L V -> L (H V)')
            # output = einsum(gather, out, 'L Z, Z E -> L E')
            # output = dropout(output, key2, 0.2)
            # orig
            # x_TxD = jax.vmap(self.ln1)(x_TxD)
            x_TxD = jax.vmap(rms_norm)(x_TxD)
            adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)
            T, D = x_TxD.shape
            C = self.n_embd // self.n_head
            # Q_TxD, K_TxD, V_TxD = jnp.split(vmap(self.c_attn)(x_TxD), 3, axis=-1)
            # Q_HxTxC = jnp.transpose(jnp.reshape(Q_TxD, (T, self.n_head, C)), (1, 0, 2))
            # K_HxTxC = jnp.transpose(jnp.reshape(K_TxD, (T, self.n_head, C)), (1, 0, 2))
            # V_HxTxC = jnp.transpose(jnp.reshape(V_TxD, (T, self.n_head, C)), (1, 0, 2))
            parts = einsum(x_TxD, self.c_attn.weight_MxN, 'L E, HsplitD E -> L HsplitD')  # note reversed order
            Q_HxTxC, K_HxTxC, V_HxTxC = rearrange(parts, 'L (split H D) -> split H L D', split=3, H=self.n_head)
            # QK LayerNorm
            # Q_HxTxC = vmap(vmap(self.q_ln))(Q_HxTxC)
            # K_HxTxC = vmap(vmap(self.k_ln))(K_HxTxC)
            Q_HxTxC = vmap(vmap(norm))(Q_HxTxC)
            K_HxTxC = vmap(vmap(norm))(K_HxTxC)
            # Rotary embeddings
            # sin_TxCp, cos_TxCp = fixed_pos_embedding(C, T)  # Cp = C//2
            # Q_HxTxC = apply_rotary_pos_emb(Q_HxTxC, sin_TxCp, cos_TxCp)
            # K_HxTxC = apply_rotary_pos_emb(K_HxTxC, sin_TxCp, cos_TxCp)

            # A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))
            # causal_mask = jnp.tril(jnp.ones((1, T, T))) == 0
            # A_HxTxT = jnp.where(causal_mask, float('-inf'), A_HxTxT)
            # orig_dtype = A_HxTxT.dtype
            # A_HxTxT = jax.nn.softmax(A_HxTxT.astype(jnp.float32) / jnp.sqrt(C), axis=-1)
            mask = jnp.tril(jnp.ones([T, T]))
            similarity = einsum(Q_HxTxC, K_HxTxC, 'H L D, H L2 D -> H L L2')
            masked_similarity = jnp.where(mask, similarity, -jnp.inf)
            orig_dtype = masked_similarity.dtype
            A_HxTxT = jax.nn.softmax(masked_similarity.astype(jnp.float32) / jnp.sqrt(C), axis=-1)
            A_HxTxT = A_HxTxT.astype(orig_dtype)

            # A_HxTxT = self.attn_dropout(A_HxTxT, inference=inference, key=adrop_key)
            # out_TxD = jnp.reshape(jnp.transpose(A_HxTxT @ V_HxTxC, (1, 0, 2)), (T, D))
            # out_TxD = self.resid_dropout(vmap(self.c_proj)(out_TxD), inference=inference, key=pdrop_key)
            A_HxTxT = dropout(A_HxTxT, adrop_key, 0.2)
            fetch = einsum(A_HxTxT, V_HxTxC, 'H L L2, H L2 V -> H L V')
            gather = rearrange(fetch, 'H L V -> L (H V)')
            output = einsum(gather, self.c_proj.weight_MxN, 'L Z, E Z -> L E')  # reversed weights
            out_TxD = dropout(output, pdrop_key, 0.2)
            return out_TxD

    class Block(eqx.Module):
        attn: CausalSelfAttention
        mlp: MLP

        def __init__(self, n_embd, n_head, dropout, key):
            key1, key2 = jrandom.split(key)
            self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout, key=key1)
            self.mlp = MLP(n_embd=n_embd, dropout=dropout, key=key2)
            init = funtree.Initializer(key2)
            # self.mlp = FunMLP(
            #    up=init.he_normal([n_embd, 4 * n_embd]),
            #    down=init.he_normal([4 * n_embd, n_embd]))

        @jax.named_scope('block')
        def __call__(self, x_TxD, inference=False, key=None):
            attn_key, mlp_key = (None, None)
            if key is not None:
                attn_key, mlp_key = jrandom.split(key)
                mlp_key_parts = jrandom.split(mlp_key, x_TxD.shape[0])
            x_TxD = x_TxD + self.attn(x_TxD, inference=inference, key=attn_key)
            # return x_TxD + self.mlp(x_TxD, mlp_key)
            return x_TxD + vmap(self.mlp)(x_TxD, mlp_key_parts)

    @ dataclass
    class GPTConfig:
        block_size: int  # Max sequence length
        vocab_size: int  # No. of tokens
        n_layer: int  # No. of transformer blocks
        n_head: int  # No. attention heads
        n_embd: int  # Hidden dimension
        dropout: float

    class GPT(eqx.Module):
        wte: Embedding
        drop: eqx.nn.Dropout
        blocks: tp.List[Block]
        ln_f: RMSNorm
        lm_head: Linear
        n_layer: int
        positional: Array

        def __init__(self, config, key):
            self.n_layer = config.n_layer
            block_key, head_key, pos_key = jrandom.split(key, 3)
            self.drop = eqx.nn.Dropout(config.dropout)
            def make_block(_key):
                attn_key, mlp_key = jr.split(_key)
                init = funtree.Initializer(mlp_key)
                if False:
                    attn = FunAttention(
                        qkv=init.he_normal([config.n_embd, config.n_embd * 3]),
                        out=init.he_normal([config.n_embd, config.n_embd]),
                        heads=config.n_head)
                    mlp = FunMLP(
                        up=init.he_normal([config.n_embd, 4 * config.n_embd]),
                        down=init.he_normal([4 * config.n_embd, config.n_embd]))
                    return FunBlock(attn=attn, mlp=mlp)
                else:
                    # attn = CausalSelfAttention(
                    #     n_embd=config.n_embd, n_head=config.n_head, dropout=config.dropout, key=attn_key)
                    # mlp = MLP(n_embd=config.n_embd, dropout=config.dropout, key=mlp_key)
                    return Block(n_embd=config.n_embd, n_head=config.n_head, dropout=config.dropout, key=mlp_key)
            self.blocks = [make_block(k) for k in jrandom.split(block_key, config.n_layer)]
            self.ln_f = RMSNorm(config.n_embd, eps=1e-5)
            embed_std = (1 / math.sqrt(config.n_embd))
            wte_wt = embed_std * jrandom.normal(head_key, (config.vocab_size, config.n_embd))
            self.wte = Embedding(config.vocab_size, config.n_embd, weight=wte_wt)
            # Share first and last layer parameters.
            # self.lm_head = Linear(config.n_embd, config.vocab_size, weight=wte_wt)
            self.lm_head = Linear(config.n_embd, config.vocab_size, key=block_key)
            self.positional = jrandom.normal(pos_key, (config.block_size, config.n_embd))

        @ jax.named_scope('gpt')
        def __call__(self, x_T, key=None):
            # Either (inference=False and key) or (inference=True and key=None)
            inference = False
            drop_key, block_keys = None, None
            if key is not None:
                drop_key, block_keys = jrandom.split(key)
                block_keys_list = jrandom.split(block_keys, self.n_layer)
            x_TxD = self.drop(self.wte(x_T), inference=inference, key=drop_key)
            x_TxD = x_TxD + self.positional[:, :]  # removeall positional
            # dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)
            # @ jax.checkpoint
            # def block_fn(_x_TxD: Array, block_and_key: tp.Tuple[GPT, tp.Optional[KeyArray]]):
            #     _dynamic_block, _key = block_and_key
            #     block = eqx.combine(_dynamic_block, static_blocks)
            #     return block(_x_TxD, inference=inference, key=_key), None
            # # Set unroll=self.n_layer for better speed (but slower compile).
            # x_TxD, _ = jax.lax.scan(block_fn, x_TxD, (dynamic_blocks, block_keys_list), unroll=1)
            for block, key in utils.zipkey(self.blocks, block_keys):
                x_TxD = block(x_TxD, key=key)
            x_TxD = vmap(self.ln_f)(x_TxD)
            logits_TxV = vmap(self.lm_head)(x_TxD)
            return logits_TxV

    @funtree.makefun
    def FunMLP(x, key, up, down):
        x_norm = jax.vmap(rms_norm)(x)
        expanded = jax.nn.gelu(einsum(x_norm, up, 'L E, E U -> L U'))
        lowered = einsum(expanded, down, 'L U, U E -> L E')
        return dropout(lowered, key, 0.2)

    @funtree.makefun
    def FunAttention(x, key, qkv, out, inference: bool, heads: int):
        x_norm = vmap(rms_norm)(x)
        parts = einsum(x_norm, qkv, 'L E, E HsplitD -> L HsplitD')
        k, q, v = rearrange(parts, 'L (split H D) -> split H L D', split=3, H=heads)
        q, k = vmap(vmap(norm))(q), vmap(vmap(norm))(k)
        H, L, D = k.shape
        mask = jnp.tril(jnp.ones([L, L]))
        similarity = einsum(k, q, 'H L D, H L2 D -> H L L2')
        masked_similarity = jnp.where(mask, similarity, -jnp.inf)
        attention = jax.nn.softmax(masked_similarity.astype(jnp.float32) / jnp.sqrt(D), axis=-1)
        attention = attention.astype(jnp.bfloat16)
        key1, key2 = jr.split(key)
        attention = dropout(attention, key1, 0.2)
        fetch = einsum(attention, v, 'H L L2, H L2 V -> H L V')
        gather = rearrange(fetch, 'H L V -> L (H V)')
        output = einsum(gather, out, 'L Z, Z E -> L E')
        output = dropout(output, key2, 0.2)
        return output

    @funtree.makefun
    def FunBlock(x, key, attn, mlp):
        attn_key, mlp_key = jr.split(key)
        x = x + attn(x, inference=False, key=attn_key)
        x = x + mlp(x, key=mlp_key)
        return x

    @funtree.makefun
    def FunGPT(x, key, embedding, positional, layers, unembed):
        L = x.shape[0]
        hidden = embedding[x] + positional[:L, :]
        for layer, k in utils.zipkey(layers, key):
            hidden = hidden + layer(hidden, key=k)
        logits = einsum(unembed, hidden, 'E O, L E -> L O')
        return logits

    def matrix(key, shape):
        return jax.nn.initializers.glorot_normal()(key, shape)
        # return jr.truncated_normal(key, lower=-2, upper=2, shape=shape) * jax.lax.rsqrt(1. * shape[0])

    # remove all the jax.nn stuff and manually init
    def make_fungpt(config, my_attention):
        init = funtree.Initializer(jr.PRNGKey(0))
        def make_layer(init):
            if my_attention:
                attn = FunAttention(
                    qkv=matrix(init.split(), [config.n_embd, config.n_embd * 3]),
                    out=matrix(init.split(), [config.n_embd, config.n_embd]),
                    heads=config.n_head)
            else:
                attn = CausalSelfAttention(
                    n_embd=config.n_embd, n_head=config.n_head, dropout=config.dropout, key=init.split())
            mlp = FunMLP(
                up=matrix(init.split(), [config.n_embd, 4 * config.n_embd]),
                down=matrix(init.split(), [4 * config.n_embd, config.n_embd]))
            return FunBlock(attn=attn, mlp=mlp)
        embedding = (1 / math.sqrt(config.n_embd)) * 1e-2 * jr.normal(init.split(), [config.vocab_size, config.n_embd])
        return FunGPT(
            embedding=embedding,
            positional=matrix(init.split(), [config.block_size, config.n_embd]),
            layers=init.map([make_layer] * config.n_layer),
            unembed=embedding.T)

    data = utils.run_const(lambda: jnp.load('shakespeare_char/train.npy'))
    meta = json.load(open('shakespeare_char/meta.json'))
    newline = [i for i, v in enumerate(meta['chars']) if v == '\n'][0]
    break_positions = jnp.where(jnp.logical_and(data[:-1] == newline, data[1:] == newline))[0] + 2

    def tasks(key):
        start = jr.choice(key, break_positions, shape=[batch_size])[:, None]
        offset = jnp.arange(seq_length)[None, :]
        return (data[start + offset], data[start + offset + 1])

    def accuracy_fn(batch):  # ([B, L], [B, L])
        return lambda x, y: 0
        x, y = batch
        def fn(model, key):
            keys = jr.split(key, x.shape[0])
            logits = jax.vmap(model)(x, keys)
            prediction = jnp.argmax(logits, axis=-1)  # argmax(softmax(x)) == argmax(x) - softmax preserves max
            return jnp.mean(prediction == y, axis=(0, 1))
        return jax.jit(fn)

    def loss_fn(params, static, x, y, key):  # model, [B, L], [B, L]
        model = eqx.combine(params, static)
        logits = jax.vmap(model)(x, jr.split(key, x.shape[0]))  # [B, L, C]
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    def update(key, model, opt_state):
        task_key, model_key = jr.split(key)
        x, y = tasks(task_key)
        return update_with_task(x, y, model, opt_state, model_key)

    @ eqx.filter_jit
    def update_with_task(x, y, model, opt_state, key):
        # doesn't work, probably same data is not good
        params, static = eqx.partition((model), eqx.is_array)
        params_faster = utils.cast_pytree(params, jnp.bfloat16)
        loss, grads = jax.value_and_grad(loss_fn)(params_faster, static, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, eqx.combine(params, static), opt_state

    batch_size = 64
    seq_length = 256
    accuracy_set = tasks(key=jax.random.PRNGKey(-1))
    config = GPTConfig(block_size=seq_length, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2)
    models = {
        'my': make_fungpt(config, my_attention=True),
        # 'theirs+mynorm': make_fungpt(config, my_attention=False),
        # 'puretheirs': GPT(config, jax.random.PRNGKey(0)),
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
        model, opt_state = utils.optimize(model, opt_state, update, accuracy_fn(accuracy_set), iter_count=2500)
        outputs[name] = dict(model=model, opt_state=opt_state)
    return outputs


outputs = utils.run(main, 'compare with midGPT')

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
