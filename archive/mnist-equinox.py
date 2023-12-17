import jax
import jax.numpy as jnp
import equinox as eqx
import mnist
import optax
import datetime
import einops

class Basic(eqx.Module):
  first: eqx.nn.Linear
  middle: list[eqx.nn.Linear]
  last: eqx.nn.Linear

  def __init__(self, key):
    first_key, last_key, *keys= jax.random.split(key, 5)
    N = 200
    self.first = eqx.nn.Linear(784, N, True, key=first_key)
    self.middle = [eqx.nn.Linear(N, N, True, key=k) for k in keys]
    self.last = eqx.nn.Linear(N, 10, True, key=last_key)

  def __call__(self, x):  # [L]
    x = self.first(x)
    for layer in self.middle:
      x = layer(x) + x
      x = jax.nn.relu(x) 
    x = self.last(x)
    x = jax.nn.softmax(x)
    return x

# 82%, 84%, 88%