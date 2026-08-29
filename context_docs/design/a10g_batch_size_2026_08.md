# Scaling the inference batch to the card — the A10G, August 2026

## What prompted it

The 2026-08-29 global campaign was the first to run two GPU types at once: the L40S in
`g6e.xlarge` (the primary) alongside the A10G in `g5.2xlarge` (the fallback added by PR
#159). Within ten hours the fills had logged roughly 4,100 actor deaths. The run before
it, ten fills over 23 hours on L40S only, logged **zero**.

## Where the deaths were

Every dead actor's host was resolved from the `Replacing dead actor N (was on i-...)`
line and looked up in EC2:

| instance type | dead-actor hosts |
|---|---:|
| `g5.2xlarge` (A10G) | **178** |
| `g6e.xlarge` (L40S) | **0** |

That is 178 of 178. The mechanism is in the Ray actor logs, unambiguous:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.54 GiB.
GPU 0 has a total capacity of 22.06 GiB of which 1.17 GiB is free.
Including non-PyTorch memory, this process has 20.89 GiB memory in use.
```

## The memory law, measured

The working set is activations — the checkpoint is ~0.2 GiB of the 20.9 in use — and
activations are linear in **`batch_size × (t_s2 + t_s1)`**: the number of pixels in the
sub-batch, times the two sequence lengths of the bucket it came from. Both factors matter,
and the second varies with geography while the first does not.

This does not have to be argued from the code. The forward sweep in
[`gpu-card-choice-2026_08.md`](gpu-card-choice-2026_08.md) already measured it on a real
A10G at the production `B = 7,168` in bf16, with `t_s1 = 0.4 × t_s2`:

| t_s2 | t_s1 | tokens per pixel | VRAM allocated |
|---:|---:|---:|---:|
| 8 | 3 | 11 | 0.88 GiB |
| 32 | 13 | 45 | 3.09 |
| 64 | 26 | 90 | 6.05 |
| 96 | 38 | 134 | 9.01 |
| 128 | 51 | 179 | 11.97 |
| 160 | 64 | 224 | 14.93 |
| 192 | 77 | 269 | 17.89 |
| 224 | 90 | 314 | 20.84 |

Regressed against tokens per pixel rather than against `t_s2`, that table is a straight
line to five decimal places:

```
allocated GiB  =  0.0660 × (tokens per pixel)  +  0.14      at B = 7,168
```

R² = 0.999992, largest residual 0.03 GiB — about **9.65 KiB per token**. Two independent
checks that this is the real law and not a curve fit:

* The narrow sweep in the same document put the A10G's last working rung at `t_s2 = 232`
  (325 tokens per pixel) and recorded 21.58 GiB there. The line predicts **21.58**. The
  next rung, 336 tokens per pixel, is predicted at 22.31 GiB against a 22.06 GiB card, and
  it is the rung that refused.
* The field failure above needed 20.89 + 2.54 = **23.43 GiB**. The line puts that at 353
  tokens per pixel. The deepest optical depth ever recorded in this campaign is 206, which
  buckets to 208; the deepest radar measured anywhere is 147, which buckets to 152; their
  sum is **360 tokens per pixel**, predicted at 23.89 GiB. The law reproduces a production
  out-of-memory event it was not fitted to, within 2%.

## Why the failures concentrate in particular fills

Because the batch was a constant and the sequence length was not. At `B = 7,168` the card
fills at **332 tokens per pixel**, and a fill's deepest bucket — not its average pixel — is
what has to fit: `iter_buckets(largest_first=True)` deliberately runs the deepest bucket
first, so a chunk containing any deep pixels pays its peak immediately.

The campaign's land-weighted mean is 173 tokens per pixel, comfortably under that wall, but
optical depth ranges 57–206 and radar depth ranges 66–147, and radar depth is **not** a
function of latitude (Pearson r = +0.009 against latitude band, where optical gives
+0.912). The two therefore go deep independently, and the places where both do are
geographically clustered. Those are the fills that died. Nothing about the tiles was
anomalous; they were on the far side of a threshold the batch size never knew about.

## Why the batch still needs no per-bucket term

The obvious next move, having found that memory depends on sequence length, is to make the
batch depend on the sequence length in hand — a token budget recomputed per bucket instead
of one pixel count. **That was considered and rejected, on two measurements.**

**First: the worst case is already bounded, and one fitted value clears it.** A pixel's sequence
is not open-ended. `compute_bin_keys` clips each of the two streams to
`max(num_obs_checkpoints)`, which is 256, so no bucket the sampler can build carries more
than **512 tokens per pixel** — a property of the bucketing, fixed at construction and
independent of geography. The deepest sub-batch that can ever be presented is therefore
`batch × 512`, and that is what the batch has to survive:

| | batch | deepest legal sub-batch | predicted VRAM | of the card |
|---|---:|---:|---:|---:|
| L40S, unchanged | 7,168 | 3,670,016 tokens | 33.9 GiB | 76% of 44.4 |
| A10G, before | 7,168 | 3,670,016 tokens | 33.9 GiB | **154% of 22.06** |
| **A10G, fitted** | **3,593** | **1,839,616 tokens** | **17.1 GiB** | **77% of 22.06** |

1,839,616 tokens is **79% of the 2,329,600 the A10G was measured to complete** (7,168
pixels at the 325-token rung). At the deepest depth ever actually observed, 360 tokens per
pixel, the fitted batch sits at 12.1 GiB — 55% of the card.

The table also shows why the L40S needs no change and gets none: at 76% of its card it
cannot reach an out-of-memory condition on any bucket the sampler can legally build. That
is a prediction the logs can falsify — if any chunk recorded `PERMANENTLY FAILED` on
2026-08-29 turns out to have died on a `g6e.xlarge`, this line is wrong and the derivation
needs re-opening.

Scaling by card memory holds the deepest legal sub-batch at the same fraction of every
card — 76% on the L40S, 77% on the A10G. That is not a coincidence; it is what a linear
memory law and a memory-proportional batch produce together, and it is why one ratio
generalises to rungs nobody has measured yet.

**Second: a token budget would buy only throughput, and the throughput is not there.** Its
sole advantage over the ratio is that it could hand back the full 7,168 pixels on shallow
buckets, where the fitted batch is conservative. But adaptive token-budget batching was
already measured and shelved — see the dead-end list in
[`inference_gpu_saturation_profile_2026_07.md`](inference_gpu_saturation_profile_2026_07.md):
`B = 7,168` is throughput-optimal at *every* sequence length, and the A10G runs SMACT 0.995
at its full 1710 MHz clock at these batch sizes, so it is already saturated. A per-bucket
batch law would add a second knob, a second calibration constant and a per-bucket branch in
the hot loop, in exchange for an unmeasured gain on a fallback rung the economics say we
would rather not be using at all.

### But the fit has to see everything that moves the cap

"No per-bucket term" is not "memory is the only input". The 512-token cap is a
*consequence* of two configured things, and a fit that ignored them would be safe only by
coincidence. Automated review found both, and they compound rather than compete:

* **A deeper ladder.** `num_obs_checkpoints` is a config field and
  `_normalize_obs_checkpoints` accepts any positive depths. A caller passing `(512,)` doubles
  every sub-batch's token count, and a memory-only ratio would still hand the A10G 3,593
  pixels — a 3,679,232-token sub-batch, well over the measured ceiling.
* **A packed card.** A fractional `num_gpus` deliberately puts several actors on one card
  — `FleetDemand.machines` documents and computes exactly that — and each of them sizing to
  the card's *total* memory oversubscribes it by the packing factor.

`batch_size_for_gpu` therefore takes all three, and the resulting demand on a card is
invariant:

| ladder | reservation | fitted batch | tokens demanded of the card | of the measured ceiling |
|---|---:|---:|---:|---:|
| default (max 256) | whole card | 3,593 | 1,839,616 | 79% |
| default (max 256) | 0.5 — two actors | 1,796 | 1,839,104 | 79% |
| `(512,)` | whole card | 1,796 | 1,839,104 | 79% |
| `(512,)` | 0.5 — two actors | 898 | 1,839,104 | 79% |

Same card, same demand, four configurations. The L40S at the default ladder and a whole
card still gets 7,168 unchanged, because its share already exceeds the tuned reference.

So the mechanism is real, and it is why the fix is needed. It is not a reason for the fix
to be more complicated, because the quantity it varies is capped and the capped value fits.

## What the margin rests on

Two things, both of which now fail a test rather than fail a fill:

* **The token cap itself.** `test_the_fitted_batch_holds_the_deepest_bucket_the_sampler_
  can_build` pins `fitted × 2 × max(checkpoints) × actors-per-card` against the measured
  ceiling, over all four combinations of ladder depth and reservation in the table above,
  and `test_the_unfitted_batch_does_not_hold_it` proves the bound is not vacuous by showing
  the old batch fails the same test. Dropping either the depth term or the packing term
  from the fit turns exactly the two affected rows red.
* **Segment-backed allocation.** Every VRAM figure here is *allocated* bytes. Under the
  default caching allocator the reserved pool runs well above that and strands the
  difference — 20.58 GiB reserved for an 11.97 GiB working set, measured — which is why the
  A10G's wall sits at 208 without the flag and 232 with it.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ships on `InferenceActor`'s
  `runtime_env` (merged in #154) and is now pinned by
  `test_the_actor_ships_the_segment_backed_allocator`.

## What it cost while unfixed

Nothing permanent in the first ten fills: a chunk gets three attempts before it is recorded
`PERMANENTLY FAILED`, and a scan of every ERROR record across all ten found zero. The cost
there was GPU time — ~4,100 actor restarts in ten hours, each reloading a 175 MB checkpoint
before the replacement could take work — so the 7,830 chunks/hour measured that afternoon
is a figure *including* this churn.

**It stopped being free.** Between 16:27Z and 16:54Z on 2026-08-29 one fill recorded 16
chunks `PERMANENTLY FAILED`, every one an out-of-memory on the third and final attempt.
Three attempts at the same chunk, on the same card, at the same batch size, is not three
chances: the sub-batch either fits or it does not, and the retry budget was being spent on
a deterministic refusal.

## Note for whoever merges this

It changes the `inference` package, so it moves the staging fingerprint. A campaign
resuming previously staged tiles must be dispatched with the existing
`staging_code_identity`, exactly as the 2026-08-29 restart was — see
[`staging-identity-and-resume.md`](staging-identity-and-resume.md).
