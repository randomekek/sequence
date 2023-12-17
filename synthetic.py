# %%
import jax
import jax.numpy as jnp
import dataclasses
import einops

"""
Recall task
Input: "A > B ; C > D ; A ?"
Output: "B @"
Alphabet softmax style
  a 0 1
  > 2
  ; 3
  ? 4
  @ 5
"""

@dataclasses.dataclass
class RecallTask:
  data : jax.Array # [batches, classes*(key_width+1+value_width+1)+1+key_width]
  expected: jax.Array # [batches, value_width]

class Alpha:
  ZERO = 0
  ONE = 1
  ARROW = 2
  SEMI = 3
  QUESTION = 4
  END = 5

def recall(key, batches, classes, key_width, value_width):
  k_key, v_key, shuf_key = jax.random.split(key, 3)
  chr = lambda value: jnp.full((batches, classes, 1), value)
  keys = jax.random.randint(k_key, (batches, classes, key_width), 0, Alpha.ONE+1)
  values = jax.random.randint(v_key, (batches, classes, value_width), 0, Alpha.ONE+1)
  data = jnp.concatenate((keys, chr(Alpha.ARROW), values, chr(Alpha.SEMI)), axis=2)
  data = einops.rearrange(data, 'b cls data -> b (cls data)')
  chb = lambda value: jnp.full((batches, 1), value)
  idx = jax.random.randint(shuf_key, (batches,), 0, classes)
  data = jnp.concatenate((data, keys[jnp.arange(batches), idx, :], chb(Alpha.QUESTION)), axis=1)
  expected = jnp.concatenate((values[jnp.arange(batches), idx, :], chb(Alpha.END)), axis=1)
  return RecallTask(data, expected)

def linearfit(key, batches, points, bit_width):
  p_key, x_key = jax.random.split(key, 2)
  m, b = jax.random.uniform(p_key, (batches, 2))  # scale this
  x = jax.random.uniform(x_key, (batches, points))
  y = m * x + b
  pass

key = jax.random.PRNGKey(2)
print(recall(key, 7, classes=3, key_width=4, value_width=0))