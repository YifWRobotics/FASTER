# Train pi05 FASTER on the tactile-grey yogaball dataset

End-to-end recipe for full fine-tuning `pi05_faster_yogaball_grey` on a single
NVIDIA H100 80GB, with training-time RTC + HAS (per the FASTER paper).

## 1. Prerequisites

- `uv` environment installed (`uv sync && uv pip install -e .`)
- LeRobot dataset placed at `data/YifWRobotics/May4-pi-05-Yogaball-Training-50hz-April26Data/`
- Miniconda env with FFmpeg 7 libs at `~/repos/miniconda3/envs/ffmpeg-libs/`
  (torchcodec needs `libavutil.so.59` etc.; the system has no ffmpeg).
  Create once with:
  ```bash
  /home/ara/repos/miniconda3/bin/conda create -y --override-channels \
      -c conda-forge -n ffmpeg-libs "ffmpeg=7.*"
  ```
- W&B logged in (`uv run --no-sync wandb login`). Requires `wandb>=0.22.3` to
  accept the new 86-char keys.

## 2. Compute normalization statistics

```bash
HF_LEROBOT_HOME=/home/ara/repos/FASTER/data \
LD_LIBRARY_PATH=/home/ara/repos/miniconda3/envs/ffmpeg-libs/lib \
uv run scripts/compute_norm_stats.py --config-name pi05_faster_yogaball_grey
```

Writes `assets/pi05_faster_yogaball_grey/<repo_id>/norm_stats.json` with
mean / std / q01 / q99 over the full dataset.

### Apply rotation-identity patch (recommended)

For dual-arm `[hand1 xyz (3), hand1 6D_rot (6), hand2 xyz (3), hand2 6D_rot (6)]`
actions, set `q01=-1, q99=+1` on the 12 rotation dims (indices 3-8 and 12-17)
of both `state` and `actions` so quantile norm reduces to identity there.
Position dims keep their empirical quantiles and scale to ~[-1, 1].

Justification: Diffusion Policy (Chi et al. 2023) and X-VLA both leave Zhou
6D rotation components un-normalized; no verified VLA codebase applies
`MEAN_STD` or `MIN_MAX` per-dim normalization to a 6D rotation slice.

## 3. Launch full fine-tune

Single H100 80GB requires `--batch-size 32` (the config's default 128 OOMs at
the rematerialization peak ~104 GB; bs=64 also OOMs at peak ~81 GB).

```bash
cd /home/ara/repos/FASTER && \
LD_LIBRARY_PATH=/home/ara/repos/miniconda3/envs/ffmpeg-libs/lib \
HF_LEROBOT_HOME=/home/ara/repos/FASTER/data \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
nohup uv run scripts/train.py pi05_faster_yogaball_grey \
    --exp-name=<your_experiment_name> \
    --batch-size 32 \
    --overwrite \
    > /home/ara/repos/FASTER/train_<your_experiment_name>.log 2>&1 < /dev/null &

echo "PID: $!"
```

### Env / flag rationale

| Component | Purpose |
|---|---|
| `LD_LIBRARY_PATH=.../ffmpeg-libs/lib` | Exposes conda-forge ffmpeg 7 libs to torchcodec for tactile-video decoding |
| `HF_LEROBOT_HOME=./data` | Lets lerobot resolve `repo_id=YifWRobotics/...` to the local dataset dir |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` | Caps JAX GPU memory at 90% of H100 capacity |
| `nohup ... &` + `< /dev/null` | Detaches from terminal; SSH disconnect won't kill the run |
| `--batch-size 32` | Override config default (128 OOMs on a single H100) |
| `--overwrite` | Clobbers an existing run dir with the same `--exp-name` |

### Throughput / ETA

- Steady-state ~1.5 s/step on H100 80GB
- 30k steps ≈ 12.5 h wall clock
- Saves at steps 5000, 10000, 15000, 20000, 25000, ~30000 (all kept via
  `keep_period=5000`); each checkpoint is ~43 GB on disk
  (`params/` 12 GB + `train_state/` 31 GB + assets/metadata)

## 4. Resume from a checkpoint

Drop `--overwrite`, add `--resume`, keep the same `--exp-name`:

```bash
... \
nohup uv run scripts/train.py pi05_faster_yogaball_grey \
    --exp-name=<same_experiment_name> \
    --batch-size 32 \
    --resume \
    > /home/ara/repos/FASTER/train_resume.log 2>&1 < /dev/null &
```

Reads `train_state/` from the latest local checkpoint under
`checkpoints/pi05_faster_yogaball_grey/<exp_name>/`.

## 5. Publish a checkpoint to HuggingFace

For deployment, upload only the `params/` subdir plus the `norm_stats.json`
snapshot. Mirror the openpi checkpoint asset layout so `create_trained_policy`
finds the norm_stats automatically:

```bash
# 12 GB params
uv run --no-sync huggingface-cli upload \
    <hf_namespace>/<model_repo> \
    checkpoints/pi05_faster_yogaball_grey/<exp_name>/<step>/params \
    params \
    --repo-type=model

# 3.6 KB norm_stats (patched, rotation-identity)
uv run --no-sync huggingface-cli upload \
    <hf_namespace>/<model_repo> \
    assets/pi05_faster_yogaball_grey/<repo_id>/norm_stats.json \
    assets/<repo_id>/norm_stats.json \
    --repo-type=model
```

Deploy-side load:

```python
from huggingface_hub import snapshot_download
ckpt_dir = snapshot_download("<hf_namespace>/<model_repo>")
# ckpt_dir/params + ckpt_dir/assets/<repo_id>/norm_stats.json
```

## 6. Known gotchas

- **OOM on bs=64 or bs=128.** Pi0FasterConfig + `gemma_2b` + `gemma_300m` +
  `action_horizon=50` + 3 SigLIP image streams exceeds 80 GB at bs >= 64.
  XLA rematerialization peaks at ~81 GB (bs=64) and ~104 GB (bs=128).
- **torchcodec fails without FFmpeg 4-7 shared libs.** The default torchcodec
  bundled with this venv looks for `libavutil.so.{56,57,58,59}`; conda-forge
  `ffmpeg=8` won't satisfy it (provides `.so.60`). Pin to `ffmpeg=7.*`.
- **W&B SDK < 0.22.3** rejects the new 86-char API keys with
  `ValueError: API key must be 40 characters long`. Upgrade with
  `uv pip install --upgrade "wandb>=0.22.3"`.
- **`pbar.write()` loss lines don't reach the redirected log.** The
  `Step N: loss=...` lines emitted by `train.py` only appear on W&B; the local
  `.log` file gets the `Progress on: N/30k rate:...` tqdm ticks. Pull losses
  via `wandb.Api()` if you need them locally.
- **Training loss plateau is expected.** The FASTER objective (Eq. 11 in the
  paper, mask-normalized over postfix with per-position HAS-shifted τ) is
  *not* the same loss function as vanilla pi0.5 — the numeric value is not
  directly comparable. The paper itself reports only downstream success rate,
  not loss curves. Evaluate on the robot, not on the loss number.
