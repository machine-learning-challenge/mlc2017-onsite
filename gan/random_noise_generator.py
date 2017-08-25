import numpy as np

noise_dim = 100

def generate_noise(batch_size):
  """Generates random noise that can be used for the input of generator model.
  Note that this will be a spec for the generator model when evaluating."""
  return np.random.uniform(-1., 1., size=[batch_size, noise_dim])

def get_dim():
  return noise_dim
