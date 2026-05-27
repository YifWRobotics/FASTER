"""Auxiliary future-force prediction head for Pi0Faster (Flax NNX).

Ported from the tactile-diffusion repo's `PerStepForceHead`
(diffusion_policy/diffusion_policy/model/common/force_head.py), which is used
by ViT-FMT and ViT-DiT. Architecture follows HTD ("Humanoid Touch Dreaming",
arxiv 2604.13015): a modular "dream-expert" head that reads per-step
conditioning tokens and emits a per-step force trajectory aligned with the
action chunk.

Three pool modes (selected by ``force_head_pool_mode``):
  ``mean``           cond.mean(axis=-2) -> MLP -> reshape to (B, T_a, F).
  ``last``           cond[:, -1, :]     -> MLP -> reshape.
  ``per_step_attn``  T_a learnable positional queries cross-attend to cond via
                     a small TransformerDecoder, then linear -> (B, T_a, F).
                     Recommended; aligns with HTD / ViT-FMT design.

Softplus and the SmoothL1 loss are applied at the call-site in
``Pi0Faster.compute_loss`` (mirroring the tactile-diffusion policies).
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


VALID_POOL_MODES = ("mean", "last", "per_step_attn")
VALID_LOSS_TYPES = ("mse", "smooth_l1")


class _CrossAttnDecoderBlock(nnx.Module):
    """Single Transformer decoder block: self-attn over queries -> cross-attn to
    memory -> FFN. Mirrors ``nn.TransformerDecoderLayer`` with batch_first=True.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, *, rngs: nnx.Rngs):
        self.norm_q1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.norm_q2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.norm_ff = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff_in = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.ff_out = nnx.Linear(dim_feedforward, d_model, rngs=rngs)

    def __call__(self, q: jax.Array, memory: jax.Array) -> jax.Array:
        # self-attn over queries
        x = self.norm_q1(q)
        q = q + self.self_attn(x, x, x)
        # cross-attn from queries to memory (encoder cond)
        x = self.norm_q2(q)
        q = q + self.cross_attn(x, memory, memory)
        # FFN
        x = self.norm_ff(q)
        x = self.ff_in(x)
        x = nnx.swish(x)
        x = self.ff_out(x)
        return q + x


class PerStepForceHead(nnx.Module):
    """Cross-attention force head: T_a learnable positional queries attend to a
    per-step conditioning sequence and produce a force trajectory
    ``(B, T_a, F)``.
    """

    def __init__(
        self,
        cond_dim: int,
        n_action_steps: int,
        force_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        state_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        # Learnable per-step positional queries.
        key = rngs.params()
        self.queries = nnx.Param(jax.random.normal(key, (1, n_action_steps, cond_dim)) * 0.02)
        # Optional state-token projection. When ``state_dim`` is set, the head
        # accepts a per-batch proprio vector and prepends a projected token to
        # the cross-attention memory. Lives under force_head/state_proj/* so it
        # is still covered by ``missing_regex=force_head/.*`` at load time.
        # Needed because in pi05=True + discrete_state_input=False the state
        # never enters the LLM (neither in prefix nor suffix), so prefix_out
        # alone has no proprio signal.
        if state_dim is not None:
            self.state_proj = nnx.Linear(state_dim, cond_dim, rngs=rngs)
        else:
            self.state_proj = None
        # Use string-keyed dict so flatten_dict(sep="/") produces only string
        # path components (`force_head/blocks/block_0/...`). Integer-keyed
        # children (from a Python list) break `_merge_params`'s `sep="/"` join.
        self.n_layers = n_layers
        self.blocks = nnx.Dict(
            **{
                f"block_{i}": _CrossAttnDecoderBlock(
                    d_model=cond_dim,
                    n_heads=n_heads,
                    dim_feedforward=cond_dim * 2,
                    rngs=rngs,
                )
                for i in range(n_layers)
            }
        )
        self.proj = nnx.Linear(cond_dim, force_dim, rngs=rngs)

    def __call__(self, cond: jax.Array, state: jax.Array | None = None) -> jax.Array:
        # cond: (B, n_cond_tokens, D); optional state: (B, state_dim).
        if self.state_proj is not None and state is not None:
            state_tok = self.state_proj(state)[:, None, :]              # (B, 1, D)
            cond = jnp.concatenate([state_tok, cond], axis=1)           # prepend
        b = cond.shape[0]
        q = jnp.broadcast_to(
            self.queries.value, (b, self.queries.value.shape[1], self.queries.value.shape[2])
        )
        for i in range(self.n_layers):
            q = self.blocks[f"block_{i}"](q, cond)
        return self.proj(q)


class _PooledForceHead(nnx.Module):
    """Pool cond (mean/last) -> MLP -> (B, T_a*F), caller reshapes."""

    def __init__(
        self,
        cond_dim: int,
        n_action_steps: int,
        force_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.n_action_steps = n_action_steps
        self.force_dim = force_dim
        self.fc1 = nnx.Linear(cond_dim, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, n_action_steps * force_dim, rngs=rngs)

    def __call__(self, pooled: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(pooled))
        x = self.fc2(x)
        return x.reshape(pooled.shape[0], self.n_action_steps, self.force_dim)


def build_force_head(
    pool_mode: str,
    cond_dim: int,
    n_action_steps: int,
    force_dim: int,
    *,
    n_layers: int = 2,
    n_heads: int = 4,
    state_dim: int | None = None,
    rngs: nnx.Rngs,
) -> nnx.Module:
    if pool_mode not in VALID_POOL_MODES:
        raise ValueError(f"force_head_pool_mode must be one of {VALID_POOL_MODES}, got {pool_mode!r}")
    if pool_mode == "per_step_attn":
        return PerStepForceHead(
            cond_dim=cond_dim,
            n_action_steps=n_action_steps,
            force_dim=force_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            state_dim=state_dim,
            rngs=rngs,
        )
    return _PooledForceHead(
        cond_dim=cond_dim,
        n_action_steps=n_action_steps,
        force_dim=force_dim,
        rngs=rngs,
    )


def apply_force_head(
    head: nnx.Module,
    cond: jax.Array,
    pool_mode: str,
    *,
    state: jax.Array | None = None,
) -> jax.Array:
    """Run the force head. Returns raw per-step predictions ``(B, T_a, F)``
    in the same normalization space as the supervision target. Softplus and
    SmoothL1 are applied at the call-site.

    The optional ``state`` argument is consumed only by ``per_step_attn``
    heads built with ``state_dim != None``: it is projected to ``cond_dim``
    and prepended as a single token to the cross-attention memory, giving
    the force head an explicit proprio signal even when the pi05 LLM never
    receives state.
    """
    if pool_mode == "per_step_attn":
        return head(cond, state=state)
    if pool_mode == "last":
        pooled = cond[:, -1, :]
    else:  # "mean"
        pooled = jnp.mean(cond, axis=-2)
    return head(pooled)


def smooth_l1(pred: jax.Array, target: jax.Array, beta: float = 1.0) -> jax.Array:
    diff = jnp.abs(pred - target)
    return jnp.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
