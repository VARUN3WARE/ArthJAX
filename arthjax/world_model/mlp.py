"""Functional JAX MLP for world model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit

from arthjax.config import EconomyConfig


def init_model_params(
    key: jax.random.PRNGKey,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
) -> dict:
    k1, k2, k3 = jr.split(key, 3)
    s1 = jnp.sqrt(2.0 / in_dim)
    s2 = jnp.sqrt(2.0 / hidden_dim)
    return {
        "W1": jr.normal(k1, (in_dim, hidden_dim)) * s1,
        "b1": jnp.zeros(hidden_dim),
        "W2": jr.normal(k2, (hidden_dim, hidden_dim)) * s2,
        "b2": jnp.zeros(hidden_dim),
        "W3": jr.normal(k3, (hidden_dim, out_dim)) * s2,
        "b3": jnp.zeros(out_dim),
    }


def model_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    h = jnp.dot(x, params["W1"]) + params["b1"]
    h = jnp.maximum(h, 0)
    h = jnp.dot(h, params["W2"]) + params["b2"]
    h = jnp.maximum(h, 0)
    return jnp.dot(h, params["W3"]) + params["b3"]


def model_loss(params: dict, x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((model_forward(params, x_batch) - y_batch) ** 2)


def make_train_step(cfg: EconomyConfig):
  grad_clip = cfg.world_model_grad_clip

  def train_step(params, x_batch, y_batch, lr):
      grads = jax.grad(model_loss)(params, x_batch, y_batch)
      grads = jax.tree.map(lambda g: jnp.clip(g, -grad_clip, grad_clip), grads)
      new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
      return new_params, model_loss(new_params, x_batch, y_batch)

  return jit(train_step)
