# Force-Prediction Head for Pi0Faster

Reference for the auxiliary force-prediction head added on branch `feelings-force`.
This is a **Pi0.5 adaptation** of the ViT-FMT / ViT-DiT cross-attention force
head from the `tactile_diffusion` repo. It follows the same core design
(learned per-step queries, decoder-style self-attn + cross-attn, direct force
regression with softplus + Smooth-L1), but the conditioning topology,
normalization mechanism, and loss masking are adjusted to fit Pi0.5's
prefix-LM architecture and FASTER's RTC training. The cumulative list of
adaptations vs strict parity is consolidated in caveat #9 below.

The head predicts a per-step **(action_horizon, 2)** trajectory of bimanual
palm normal force `[right, left]`, supervised end-to-end alongside the action
flow-matching loss. The pretrained `pi05_base` backbone loads unchanged; only
the new `force_head/*` params start from random init.

---

## Quick start (yogaball / bucket / pillow at 25 Hz with force supervision)

```bash
# 1) Compute normalization stats for state + actions (force is NOT in the stats;
#    see "Force normalization" below for why). The flag --config-name is required.
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_faster_yogaball_25hz_grey_force
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_faster_bucket_grey_force
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_faster_pillow_grey_force

# 2) Full fine-tune. Init from pi05_base; force_head/* starts random.
.venv/bin/python scripts/train.py pi05_faster_yogaball_25hz_grey_force --exp-name <name>
.venv/bin/python scripts/train.py pi05_faster_bucket_grey_force        --exp-name <name>
.venv/bin/python scripts/train.py pi05_faster_pillow_grey_force        --exp-name <name>
```

At deploy, post-process the model's force output to recover physical Newtons:
clip to `[0, 1]` (softplus is non-negative but unbounded — see below), then
multiply by `force_cap` (default `20.0`) before handing to the admittance
controller.

---

## Files changed / added

| File | Change |
|---|---|
| `src/openpi/models/force_head.py` (new) | Flax NNX `PerStepForceHead`: H_a learnable positional queries → 2 hand-written cross-attention decoder blocks (`_CrossAttnDecoderBlock`: self-attn → cross-attn → FFN, stacked via `nnx.Dict` with string keys to keep flatten_dict happy) → linear `D → force_dim`. Also `mean` / `last` pool-mode fallbacks and a `smooth_l1` helper. The blocks are functionally equivalent to `nn.TransformerDecoder` but implemented manually since openpi has no NNX-native transformer-decoder primitive. |
| `src/openpi/models/pi0_config.py` | Added force-head fields to `Pi0FasterConfig` (`force_head_enabled`, `force_head_dim`, `force_head_pool_mode`, `force_head_n_layers`, `force_head_n_heads`, `force_head_softplus`, `force_head_loss_type`, `force_loss_weight`). `inputs_spec` emits a `force_target` spec when enabled. All defaults preserve existing behavior. |
| `src/openpi/models/pi0_faster.py` | Builds `self.force_head` when enabled with `cond_dim=paligemma_config.width`. `compute_loss` runs the head on `prefix_out` (the **clean** obs conditioning — uncontaminated by suffix noise/time, since prefix queries can't attend to suffix keys under the prefix-LM mask) plus `observation.state` (projected into `force_head/state_proj/*` and prepended to memory, since pi05+`discrete_state_input=False` otherwise hides state from the LLM). Applies softplus, computes Smooth-L1 against `observation.force_target`, masks by `postfix_action_mask`, adds `force_loss_weight * force_loss` to the action loss. **Raises during training** if `force_head_enabled=True` and `force_target is None`. |
| `src/openpi/models/model.py` | Added `force_target` (optional) to the `Observation` dataclass; threaded through `from_dict` and `preprocess_observation`. |
| `src/openpi/training/weight_loaders.py` | `CheckpointWeightLoader.missing_regex` is now a configurable field (default still `.*lora.*` so every other config is unaffected). |
| `src/openpi/training/config.py` | Added `TactileForceInputs` and `TactileForceDataConfig` (data pipeline for the force-instrumented datasets). Added three new `TrainConfig`s: `pi05_faster_yogaball_25hz_grey_force`, `pi05_faster_bucket_grey_force`, `pi05_faster_pillow_grey_force`. **The existing non-force configs are untouched.** |

---

## Architecture

```
  observation (images, state, prompt)
        │
        ▼
  PaliGemma prefix tokens ────► shared LLM ◄──── action-expert suffix tokens (noisy actions + time)
        │                          │                         │
        ▼                          ▼                         ▼
  (prefix_out)              (suffix_out)              (per-step action embeddings)
        │                          │
        │                          ├──── action_out_proj ────► v_t (flow-matching velocity)
        │                          │                                        │
        │                          │                                        ▼
        │                          │                              MSE loss vs (noise - actions)
        │                          │
        │                          (NOT used by force head; suffix is noise+time-conditioned)
        ▼
  PerStepForceHead ────► raw force_pred (B, ah, 2), unbounded
        │
        ▼
  softplus (non-negative, but softplus(0) ≈ 0.693 — see "Force normalization")
        │
        ▼
  Smooth-L1 vs force_target (B, ah, 2) in [0, 1]
        │
        ▼
  λ_F × force_loss
```

### Force head module (`PerStepForceHead`)

- `H_a` learnable positional queries `Q ∈ R^{1 × ah × D}` (D = `paligemma_config.width`, 2048 for `gemma_2b`).
- Optional `state_proj`: `nn.Linear(action_dim → D)` projecting `observation.state` into a single token prepended to the cross-attention memory. Active when `force_head_use_state=True` (default). All params land under `force_head/state_proj/*`.
- 2 hand-written cross-attention decoder blocks (`_CrossAttnDecoderBlock` in `force_head.py`): self-attn over queries → cross-attn (queries → `[state_token] ++ prefix_out`) → FFN. Functionally equivalent to `nn.TransformerDecoder` but written natively in Flax NNX and stacked via `nnx.Dict` (string-keyed) so `_merge_params`'s `flatten_dict(sep="/")` doesn't trip on integer-keyed list children.
- 4 attention heads.
- Linear `D → force_dim=2` projection.

**KV source**: `[state_token] ++ prefix_out`. Why both:

1. **`prefix_out`** is the LLM output at the PaliGemma image+text token positions. This is the **clean** observation conditioning — the prefix-LM attention mask blocks prefix queries from attending to suffix keys, so `prefix_out` is uncontaminated by the diffusion noise schedule, action noise, or time embedding. This matches the memory source ViT-FMT/DiT use for their force head (the encoder output `cond`, before the action denoiser).

2. **`state_token`** is a single projected token from `observation.state`. Necessary because in our configs (`pi05=True`, `discrete_state_input=False`) the LLM never sees state directly: `TokenizePrompt` skips it when `discrete_state_input=False`, and `embed_suffix`'s state-token branch only runs when `not self.pi05`. So `prefix_out` carries only image + text features — the force head would have no proprio signal without the projected state token.

> **Adaptation, not exact parity.** ViT-FMT/DiT condition the force head on a *shared* obs encoder output that already mixes tactile + hand pose features. FASTER's state path is **local to the force head** (`force_head/state_proj/*`), so the obs encoder (PaliGemma + SigLIP) is not jointly updated by the state projection's gradient — only by the cross-attention path through `prefix_out`. The functional information available to the force head is the same (tactile + proprio), but the gradient topology differs from a strict ViT-FMT/DiT port. Treat this as a Pi0.5 adaptation, not a byte-for-byte clone.

History: an earlier version of this head read from `suffix_out` (noise/time-conditioned) and skipped the state projection entirely. Both bugs are now fixed.

Gradients flow back through `prefix_out` into the shared PaliGemma encoder
(and SigLIP image encoder upstream), so the obs encoder is jointly updated by
both the action and force losses. The state path's gradient stays inside
`force_head/state_proj/*`.

---

## Data pipeline

### Source datasets

Three force-instrumented LeRobot datasets, all at 25 Hz, expected at
`$HF_LEROBOT_HOME` (typically `/home/yifan/Robotics/openpi-IsaacLab/third_party/lerobot_datasets`):

- `mixed_yogaball_force_25hz_lerobot`
- `mixed_bucket_force_25hz_lerobot`
- `mixed_pillow_force_25hz_lerobot`

**Column layout:**
```
action[18]            = [left_pose9, right_pose9]                 (pose target, no force)
observation.state[20] = [left_pose9, right_pose9, right_force, left_force]
                         └──────── 0:18 ────────┘  └──── 18:20 ────┘
observation.images.{chest,left,right}_tactile  (channel-replicated greyscale)
```

### `TactileForceInputs`

The new transform requests `observation.state` as a **50-step sequence**
(`action_sequence_keys=("action", "observation.state")`), then splits:

- `inputs["state"]`        = `state_seq[0, :18]`              → current pose (18,), matches the existing 18-dim convention so pretrained Pi0.5 weights behave identically.
- `inputs["force_target"]` = `clip(state_seq[:, 18:20], 0, force_cap) / force_cap`  → `(50, 2)` in `[0, 1]`.

### Force normalization (the 20 N cap)

**This is NOT a faithful port of ViT-FMT/DiT's normalization.** ViT-FMT/DiT
fits `force_target` in its `LinearNormalizer` along with every other obs key,
then applies softplus *in physical Newton space* after unnormalizing the model
output, and re-normalizes the result for the loss (see
`diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`,
`_maybe_compute_force_loss` in the `tactile_diffusion` repo).
That mechanism requires the model to have access to the normalizer's stats —
which FASTER's model doesn't, since normalization in openpi lives in the data
pipeline, not in the model.

What FASTER does instead is a simpler **fixed-cap pre-scaling** in
`TactileForceInputs`:

```python
force_target = np.clip(force_raw_newtons, 0.0, force_cap) / force_cap   # [0, 1]
```

We chose this for two reasons:

1. **Softplus + openpi's quantile normalization are incompatible.** FASTER uses
   quantile normalization for Pi0.5 (`q01/q99 → [-1, 1]`). For a non-negative
   quantity like force, `0 N` maps near `-1` in normalized space. But softplus
   is strictly positive, so low-force targets become structurally hard to fit
   (the model has to learn to produce strongly negative logits, and softplus
   compresses the gradient there).
2. **Scale matching.** Action loss runs in `[-1, 1]` quantile space (roughly
   unit scale). Raw force in Newtons is `O(10)`. Without scaling, `force_loss`
   would dominate `action_loss` by ~10–30×.

The 20 N cap is borrowed from the **paper-level convention** in ViT-FMT/DiT
(`f_max = 20`), but the implementation mechanism (fixed pre-scaling vs
quantile-fit normalizer + softplus-in-physical-space) is different. If you want
a tighter match, you'd need to plumb `force_target` stats into `Pi0Faster` and
do the unnormalize → softplus → re-normalize round-trip in `compute_loss`
(~30–50 LOC, separate ablation).

**Softplus caveats** (we still apply softplus on the raw model output for
non-negativity, but the math isn't perfect for the [0, 1] target):

- `softplus(0) ≈ 0.693`, **not** `0`. So even at the "natural" zero of the
  pre-softplus output, the predicted (post-softplus) value is `~0.693`, which
  in the 20 N cap space is `~13.8 N`. To predict near-zero force, the model
  must learn strongly negative pre-softplus logits (softplus(−5) ≈ 0.007).
  This is learnable but biases predictions slightly away from exact zero.
- Softplus is non-negative but **unbounded**. The model output is not
  intrinsically constrained to `[0, 1]`. The Smooth-L1 loss against the
  `[0, 1]` target is what shapes the prediction range. At inference, clip the
  output to `[0, 1]` before scaling by `force_cap`.

If exact zero matters for your downstream controller, consider replacing
softplus with ReLU (`force_head_softplus=False` and apply `nnx.relu` instead),
or removing the nonlinearity entirely and clipping at inference. Both are
trivial edits to `compute_loss`.

At inference:
```python
force_pred_newtons = np.clip(model_output, 0.0, 1.0) * force_cap
```

---

## Weight loading (preserves pretrained backbone)

`CheckpointWeightLoader.missing_regex` is the configurable hook that allows new
params (here, `force_head/*`) to be absent from a pretrained checkpoint and kept
at their freshly initialized values.

In the three force configs:
```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_base/params",
    missing_regex=r"force_head/.*",
),
```

What happens at load:
- `_merge_params(checkpoint, target, missing_regex=r"force_head/.*")`:
  - For every key present in both checkpoint and model: take the checkpoint value.
  - For every key matching `force_head/.*` in the model but absent from the
    checkpoint: keep the fresh init.
  - For anything else missing or extra: error.
- `check_pytree_equality(expected=model_state, got=merged, check_shapes=True)`
  passes because every model key is now accounted for.

Net: backbone (PaliGemma + action expert + action projection) loads from
`pi05_base`. `force_head/*` (~57 params: queries, 2× decoder blocks, projection,
plus `state_proj/{kernel,bias}` when `force_head_use_state=True`) stays at fresh
init. Full fine-tune — no LoRA, no frozen weights.

This regex is NOT a LoRA mechanism. The default `.*lora.*` exists so LoRA
configs in the repo can be loaded the same way; for full fine-tune + force head,
the regex covers only the new head's namespace.

---

## Config knobs

In `Pi0FasterConfig`:

| Field | Default | Notes |
|---|---|---|
| `force_head_enabled` | `False` | Master switch. When `False`, no `force_head` submodule is created and the param tree is byte-identical to the pre-existing model. |
| `force_head_dim` | `2` | `[right_force, left_force]`. |
| `force_head_pool_mode` | `"per_step_attn"` | `mean` / `last` / `per_step_attn`. Only the last one is a real cross-attention head. |
| `force_head_n_layers` | `2` | Decoder layers (self-attn → cross-attn → FFN). |
| `force_head_n_heads` | `4` | Attention heads. |
| `force_head_softplus` | `True` | Applies softplus to the raw model output. Softplus is non-negative but unbounded; `softplus(0) ≈ 0.693`. See "Force normalization" for the trade-offs against the `[0, 1]` capped target. Set to `False` if you want to clip externally or replace with ReLU. |
| `force_head_use_state` | `True` | When True, the force head receives `observation.state` and projects it (`force_head/state_proj/*`, shape `action_dim → cond_dim`) into an extra cross-attention memory token. Required for pi05+`discrete_state_input=False` configs where the LLM otherwise hides state from `prefix_out`. Set to `False` only if you intentionally want a vision-only force head. |
| `force_head_loss_type` | `"smooth_l1"` | `mse` or `smooth_l1`. |
| `force_loss_weight` | `1.0` | λ_F. With the 20 N cap the action and force losses are scale-matched; 1.0 is a reasonable starting point. |

In `TactileForceDataConfig`:

| Field | Default | Notes |
|---|---|---|
| `force_cap` | `20.0` | Saturation point for raw Newton force before division. Per-task override; e.g. lower for the pillow if peak forces stay small. |

---

## Caveats / gotchas

1. **Do not resume an old non-force checkpoint with `--resume` and force on.**
   Resume restores the full TrainState (params + optimizer state + EMA), all of
   which assume a fixed pytree shape. Use `weight_loader=CheckpointWeightLoader(...)`
   on a **fresh run** instead. (EMA is initialized from the fresh params at
   step 0, so it gets `force_head/*` correctly.)

2. **`action_dim` must stay `32`** and the model variants must stay
   `gemma_2b` / `gemma_300m`. These determine pretrained param shapes; the
   shape-checking equality test at `train.py:76` will reject mismatches.

3. **Action horizon stays `50`** for the same reason. At 25 Hz this is a 2-sec
   prediction window for both the action chunk and the force trajectory.

4. **Pad masking.** Episodes shorter than `action_horizon` get zero-padded;
   LeRobot exposes `action_is_pad` and `observation.state_is_pad`. Neither the
   action loss nor the force loss currently masks these pads (only the RTC
   `postfix_action_mask` for the delay prefix is applied). If you want strict
   pad-aware loss, that's a separate change — applies to both action and force
   losses symmetrically.

5. **Silent skip is removed.** If `force_head_enabled=True` and the batch
   doesn't carry `force_target`, `compute_loss` **raises** during training.
   Inference (`train=False`) still allows missing force_target so callers can
   sample actions without supplying force.

6. **`compute_norm_stats` doesn't touch force.** It only computes stats for
   `state` and `actions` (hardcoded keys list in `scripts/compute_norm_stats.py`).
   `force_target` is already pre-scaled to `[0, 1]` by `TactileForceInputs`, so
   the `Normalize` transform's `strict=False` fallthrough passes it through
   unchanged — which is exactly what we want. The script's CLI requires the
   `--config-name` flag (it is not positional).

7. **Force head reads `prefix_out`, not `suffix_out`.** This is the same
   "clean obs cond" memory source that ViT-FMT/DiT use. An earlier
   implementation read `suffix_out` (the action-expert / noisy-action stream),
   which would have forced the head to disentangle the noise schedule. If you
   port more pieces from `tactile_diffusion`, keep this in mind: the obs
   encoder output in pi0 is `prefix_out`, not `suffix_out`.

8. **Quantile normalization of pose dims breaks SO(3) geometry.** Existing
   openpi behavior, not force-head-specific. The 18-dim hand pose is laid
   out as `[left_pose9, right_pose9]` where each 9-tuple is `[xyz, R_6d]`
   (or similar). `compute_norm_stats` fits per-dimension quantile stats and
   then normalizes each column independently to `[-1, 1]`. The raw rotation
   columns satisfy a geometric constraint (e.g. the 6D-rotation columns lie
   on a 2-frame manifold), but after independent per-axis quantile rescaling
   the normalized training tensors no longer preserve that constraint.
   This affects both action and state inputs — including the state token
   the force head consumes via `state_proj`. We do not currently fix this,
   because the existing yogaball/bucket/pillow training also operates under
   this convention and it has not been a problem in practice. If you ever
   move to a rotation-aware loss or constraint, the projection from raw
   pose to model input would need a redesign upstream of `compute_norm_stats`.

9. **Force loss masking is FASTER-specific, not a ViT-FMT/DiT port.** The
   force loss in `pi0_faster.compute_loss` is masked by `postfix_action_mask`
   — the same mask the action loss uses to skip the RTC prefix region
   (positions `< delay` get ground-truth action prefixes, not predictions, so
   they don't get supervised). ViT-FMT/DiT have no RTC delay concept; their
   force loss is averaged uniformly over the full prediction horizon. Sharing
   the mask is defensible (the head shouldn't be supervised at positions where
   the action denoiser isn't being supervised either), but it is another
   reason this implementation is a **Pi0.5 adaptation**, not a byte-for-byte
   ViT-FMT/DiT port. The cumulative list of adaptations vs strict parity:
   (a) state path is local to `force_head/state_proj/*` rather than the
       shared obs encoder,
   (b) force normalization is a fixed 20 N cap rather than a fitted
       LinearNormalizer + unnormalize→softplus→re-normalize round-trip,
   (c) force loss is RTC-prefix-masked rather than full-horizon-averaged.

---

## Verifying after a code change

The smallest-viable check that the head still installs and loads pretrained
weights cleanly:

```bash
.venv/bin/python -c "
import jax, flax.nnx as nnx, flax.traverse_util as tu
from openpi.models import pi0_config
from openpi.training import weight_loaders as wl
from openpi.shared import array_typing as at

cfg_off = pi0_config.Pi0FasterConfig(pi05=True, action_dim=32, action_horizon=8,
                                     max_token_len=16, force_head_enabled=False)
cfg_on  = pi0_config.Pi0FasterConfig(pi05=True, action_dim=32, action_horizon=8,
                                     max_token_len=16, force_head_enabled=True,
                                     force_head_dim=2)

def shape_tree(cfg):
    return nnx.split(nnx.eval_shape(cfg.create, jax.random.key(0)))[1].to_pure_dict()

pretrained = shape_tree(cfg_off)
target     = shape_tree(cfg_on)
merged     = wl._merge_params(pretrained, target, missing_regex=r'force_head/.*')
at.check_pytree_equality(expected=target, got=merged, check_shapes=True, check_dtypes=False)
print('PASS — pretrained backbone loads, force_head/* stays fresh init')
"
```

And the data pipeline end-to-end test (requires the LeRobot datasets to be
locally available under `$HF_LEROBOT_HOME`):

```bash
HF_LEROBOT_HOME=/home/yifan/Robotics/openpi-IsaacLab/third_party/lerobot_datasets \
.venv/bin/python -c "
import pathlib
from openpi.training import config as _cfg, data_loader as _dl
train_cfg = _cfg.get_config('pi05_faster_yogaball_25hz_grey_force')
data_cfg = train_cfg.data.create(pathlib.Path('/tmp'), train_cfg.model)
ds = _dl.create_torch_dataset(data_cfg, train_cfg.model.action_horizon, train_cfg.model)
s = ds[0]
s = data_cfg.repack_transforms.inputs[0](s)
s = data_cfg.data_transforms.inputs[0](s)
print('force_target:', s['force_target'].shape, s['force_target'].dtype,
      'range:', float(s['force_target'].min()), float(s['force_target'].max()))
"
```

Expected: `force_target: (50, 2) float32  range: 0.0 <some value <= 1.0>`.
