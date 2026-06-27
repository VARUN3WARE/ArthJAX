"""Latent encoder for world model v2 (optional compression path)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr


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
