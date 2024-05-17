"""
validate TDLN space via interpolating two digits


"""

def code():
  a = test_img[1]
  b = test_img[4]
  def m(c):
    return b + (a-b)*c
  mid = jax.vmap(m)(jnp.arange(start=-0.5, stop=1.5, step=0.1))
  plt.figure(figsize = (3, 30))
  plt.imshow(jnp.vstack(jax.vmap(final_model)(mid).reshape(-1, 28)))
