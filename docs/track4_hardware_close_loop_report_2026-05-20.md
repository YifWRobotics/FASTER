# Track 4 (async FASTER) — Hardware Close-Loop Report

**Run timestamp:** `20260520_002732` (lab PC, RTX Pro 6000)
**Capture:** `closed_loop_snap_debug.zip` (Desktop)
**Outcome:** _Pipeline validated end-to-end on hardware. Visibly successful task progression. Two follow-ups flagged below._

---

## 1. Summary

| Metric                            | Value                                  |
| --------------------------------- | -------------------------------------- |
| Run duration                       | 35.7 s                                 |
| Ticks published                    | 883                                    |
| Chunks (inferences)                | 69                                     |
| Inference latency (chunks 1+)      | mean 71 ms · median 67 ms · p95 89 ms · max 106 ms |
| Hard-inpaint max error             | **5.96e-8** (float32 rounding noise — bit-exact) |
| Latency overruns                   | 4 / 69 = 5.8 %                         |
| Chunk-boundary snap (LH/RH)        | 5.4 / 5.5 mm — basically identical to within-chunk Δ (4.5 / 4.6 mm) |
| Effective publish rate             | 24.7 Hz vs 50 Hz target ⚠               |
| Max inter-tick stall               | 395 ms (~20-tick blackout) ⚠            |

## 2. Architecture validation

### Hard-inpaint (the core of Track 4)
- `faster_inpaint_max_err` across all 69 chunks: **5.96e-8** worst case (float32 ULP territory)
- `faster_inpaint_mean_err`: mean 1.56e-8, max 2.12e-8
- Verdict: **FASTER server's per-Euler-step hard-inpaint is working bit-exactly.** The first `delay_used` rows of every returned chunk equal the `action_prefix` we sent in.

### Self-prefix seed (chunk 1, `delay=14`)
- Std across the 14 tiled rows: 1.19e-7 per dim (float32 noise)
- Verdict: self-prefix tile of `obs.state` survived the websocket round-trip unchanged. No template-collapse risk on the seed inference.

### Dynamic delay tracker
- `delay_sent` distribution across 69 chunks: `{4: 6, 5: 51, 6: 2, 14: 10}`
- `new_delay` (actual observed): `{4: 49, 5: 17, 6: 2, 7: 1}`
- The 10 × `delay=14` are early chunks where the rolling-max latency tracker was still seeded; once latency was observed it settled at 4–5.
- Verdict: tracker self-tunes correctly. Same behavior as the open-loop hardware test on the same GPU.

### Inference latency (post-warmup)
- mean 71 ms, median 67 ms, p95 89 ms, max 106 ms
- Chunk 0 = 101 ms — **no JIT spike** (server was warmed up before T2 launched, as recommended)

## 3. Issues found

### 3.1 Latency overruns: 4 / 69 chunks (5.8 %)
Chunks 16, 28, 30, 61 all sent `delay=4` but actual `new_delay` was 5 (or 7 on chunk 61, infer=85 ms). The rolling-max tracker undershoots when latency creeps up after a quiet stretch.

**Impact:** None observed. Hard-inpaint clamps the prefix to the last commanded action even on overrun, so the actor still has a continuous trajectory — overrun just means it briefly lands on a postfix position that wasn't anchored to its "frozen past".

**Fix:** Tighten the tracker floor.
```python
# In _LatencyTracker — replace
delay_used = max(1, min(FASTER_MAX_DELAY, ...))
# with
delay_used = max(5, min(FASTER_MAX_DELAY, ...))
```
Cheap and safe given measured infer p95 = 89 ms ≈ 4.5 ticks at 50 Hz.

### 3.2 Publish-rate irregularity (the main thing to chase)

**Symptoms:**
- 883 ticks / 35.7 s = **24.7 Hz effective vs 50 Hz target**
- median dt = 27.6 ms (close to 20 ms target — most ticks are healthy)
- mean dt = 40.5 ms — average is dragged up by tail
- **max dt = 395 ms** (~20 ticks of blackout in one event)

**Why this is the actor's publish loop, not inference:**
- Inference is in a worker thread; `_tick` only acquires `_infer_lock` briefly to read the trajectory
- The lock is released BEFORE the heavy post-chunk work (logging, npz dump)
- So tick stalls cannot come from inference latency directly — they come from CPU-bound Python work elsewhere stealing the GIL from the tick thread

**Most likely root causes (in order of suspicion):**

1. **GIL contention from per-chunk worker post-processing.** After each chunk arrives, the worker thread does (in order, holding GIL the whole time):
   - msgpack deserialization of the websocket response
   - Snap-back diagnostic computation (numpy norms over chunk)
   - **A 14-line `get_logger().info()` with ~40 formatted floats and 6 numpy `np.round(..., 3).tolist()` calls** — this is heavy and flushes to stdout
   - **`np.savez_compressed(...)` of ~10 KB of arrays** — zlib at default level 6 can take 20–50 ms
   - Even though `_tick` is on a separate `MutuallyExclusiveCallbackGroup` and the executor has 4 threads, **Python threads do not parallelize CPU work** — the timer's wakeup still has to wait for the worker to release GIL

2. **CSV row appending in `_tick`.** Every tick writes one row to `per_step.csv`. If unbuffered and the disk is slow, this can stall sporadically.

3. **PoseStamped publish into the IK chain.** If the downstream Pink IK or controller pipeline is QoS-blocking, the actor's `publish()` call returns slowly.

4. **Python GC pauses.** Each chunk allocates ~50×18 trajectory arrays + tactile buffers; long-tail generational collection can pause for tens of ms.

**Debug plan (cheapest first):**

```python
# Add to _tick(), top of method:
def _tick(self) -> None:
    t_now = self._now_seconds()
    if hasattr(self, "_prev_tick_t"):
        gap = t_now - self._prev_tick_t
        if gap > 0.050:  # 2.5x target period
            self.get_logger().warn(
                f"[tick stall] {gap*1000:.0f}ms — last chunk_id={self._chunk_id} "
                f"in_flight={self._infer_in_flight}"
            )
    self._prev_tick_t = t_now
    ...
```

This will tell us WHEN stalls happen (correlation with chunk boundaries is the GIL-contention smoking gun) and HOW LONG each one is.

Then time the heavy blocks individually:
```python
# Around the per-chunk log line:
_t0 = time.perf_counter()
self.get_logger().info(...)  # the 14-line log
_t_log = (time.perf_counter() - _t0) * 1000.0
_t0 = time.perf_counter()
np.savez_compressed(...)
_t_dump = (time.perf_counter() - _t0) * 1000.0
self.get_logger().info(f"[post-chunk timing] log={_t_log:.0f}ms dump={_t_dump:.0f}ms")
```

**Fixes (in order of likely impact-per-effort):**

1. **Move npz dump + log formatting off the worker thread.** Push the dump payload into a `queue.Queue`; have a daemon dumper thread `np.savez_compressed` from the queue. Tick contends with a sleeping dumper, not a CPU-bound one. (~30 lines of code; biggest win.)

2. **Switch to `np.savez` (uncompressed) — or `compresslevel=1`.** Closed-loop dumps are ~10 KB; even uncompressed at 50 dumps/run = 500 KB. Trade a little disk for ~20 ms per chunk of CPU. Easy:
   ```python
   np.savez(fpath, ...)   # instead of savez_compressed
   ```

3. **Quiet the per-chunk logger.info()** — collapse the 14-line snap-back diagnostic into one short line in the live log, dump the full multi-line detail to the npz only. Saves a stdout flush burst per chunk.

4. **Buffer the CSV writer.** Open with `buffering=1<<16` and only flush on shutdown / every N ticks. (If we can confirm CSV is part of the cause via the timing hooks above.)

5. **Pin the tick to a real-time scheduling class.** `os.sched_setscheduler(tid, SCHED_FIFO, ...)` for the timer thread. This is a sledgehammer; do it only if (1)–(4) don't fix the tail.

### 3.3 Tracking error (state vs cmd): LH 84 mm mean, RH 66 mm mean

This is **controller-side lag**, not policy snap:
- Within-chunk per-tick command delta = 4.5 mm
- Chunk-boundary command delta = 5.4 mm
- These are basically equal → temporal_ensemble is hiding all the chunk seams, no policy-level snap

The state-vs-command gap reflects the IK/joint-controller's finite gain plus contact dynamics with the yoga ball. Independent of the Track 4 work.

## 4. Verdict

Track 4 closed-loop is working on hardware. The architecture choices (hard-inpaint, t_dispatch anchoring, self-prefix seed, rolling-max delay tracker, temporal ensembling) all validate against the data.

Two follow-ups before longer / harder runs:
- Bump tracker floor 1 → 5 (kills the 5.8 % overrun rate)
- Diagnose + fix the 395 ms publish stall (likely GIL contention from per-chunk worker post-processing — move npz dump + heavy logging off the worker thread)
