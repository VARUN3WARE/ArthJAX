"""Latent encoder for world model full-state compression."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit

from arthjax.config import EconomyConfig
from arthjax.world_model.mlp import init_model_params, model_forward


def init_encoder_params(
    key: jax.random.PRNGKey,
    in_dim: int,
    latent_dim: int,
) -> dict:
    k1, k2 = jr.split(key)
    s = jnp.sqrt(2.0 / in_dim)
    return {
        "We": jr.normal(k1, (in_dim, latent_dim)) * s,
        "be": jnp.zeros(latent_dim),
        "Wd": jr.normal(k2, (latent_dim, in_dim)) * jnp.sqrt(2.0 / latent_dim),
        "bd": jnp.zeros(in_dim),
    }


def encode(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    h = jnp.dot(x, params["We"]) + params["be"]
    return jnp.maximum(h, 0)


def decode(params: dict, z: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(z, params["Wd"]) + params["bd"]


def latent_loss(
    encoder: dict,
    mlp: dict,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    recon_weight: float = 0.05,
) -> jnp.ndarray:
    """Predict next latent state with optional reconstruction regularizer."""
    z = encode(encoder, x_batch)
    z_tgt = encode(encoder, y_batch)
    z_pred = model_forward(mlp, z)
    latent_mse = jnp.mean((z_pred - z_tgt) ** 2)
    recon_mse = jnp.mean((decode(encoder, z_pred) - y_batch) ** 2)
    return latent_mse + recon_weight * recon_mse


def make_latent_train_step(cfg: EconomyConfig):
    grad_clip = cfg.world_model_grad_clip
    recon_weight = 0.05

    def train_step(encoder, mlp, x_batch, y_batch, lr):
        def loss_fn(enc, net, xb, yb):
            return latent_loss(enc, net, xb, yb, recon_weight)

        grads_enc, grads_mlp = jax.grad(loss_fn, argnums=(0, 1))(
            encoder, mlp, x_batch, y_batch
        )
        grads_enc = jax.tree.map(lambda g: jnp.clip(g, -grad_clip, grad_clip), grads_enc)
        grads_mlp = jax.tree.map(lambda g: jnp.clip(g, -grad_clip, grad_clip), grads_mlp)
        new_enc = jax.tree.map(lambda p, g: p - lr * g, encoder, grads_enc)
        new_mlp = jax.tree.map(lambda p, g: p - lr * g, mlp, grads_mlp)
        return new_enc, new_mlp, loss_fn(new_enc, new_mlp, x_batch, y_batch)

    return jit(train_step)


def init_latent_world_model(
    key: jax.random.PRNGKey,
    in_dim: int,
    latent_dim: int,
    hidden_dim: int,
) -> tuple[dict, dict]:
    """Encoder (in_dim→latent) + MLP (latent→latent)."""
    key, k_enc, k_mlp = jr.split(key, 3)
    encoder = init_encoder_params(k_enc, in_dim, latent_dim)
    mlp = init_model_params(k_mlp, latent_dim, hidden_dim, latent_dim)
    return encoder, mlp


def latent_predict_next(encoder: dict, mlp: dict, x_norm: jnp.ndarray) -> jnp.ndarray:
    """One latent-space transition in normalized coordinates."""
    z = encode(encoder, x_norm)
    z_next = model_forward(mlp, z)
    return decode(encoder, z_next)
