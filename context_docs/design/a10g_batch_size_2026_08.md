# Why the inference batch is sized to the card it lands on

**August 2026.**

**Audience:** anyone who wants to understand this change, including people who have never
worked on this pipeline. Every term is defined where it first appears, and the code
locations are collected in an appendix so the explanation does not depend on them.

**One-paragraph summary.** The pipeline runs a neural network on satellite imagery, on
rented graphics cards. It had one batch size — how much work it hands the card at a time —
tuned for a 44 GiB card. On 29 August 2026 the campaign began renting a second, smaller
22 GiB card as well, and on that card the same batch did not fit: about 4,100 worker
processes were killed by out-of-memory errors in ten hours, and 16 units of work were
abandoned permanently. The fix is to compute the batch from the card's memory rather than
using a fixed number. This document explains the mechanism, gives the measurements it rests
on, and sets out the three alternatives that were considered and why each was rejected.

---

## 1. Background: what the machine is actually doing

Five terms carry the whole argument.

**Pixel.** One ground location, 10 m square, with one year of satellite history attached to
it. The network's job is to turn that history into a short numeric summary.

**Observation, and depth.** One usable satellite pass over that pixel. There are two
independent streams — **optical** (Sentinel-2, blocked by cloud) and **radar**
(Sentinel-1, not blocked by cloud) — and each pixel has its own count of each. A pixel's
**depth** is how many observations it carries. Depth varies enormously across the globe: a
cloudy tropical pixel may have 57 usable optical passes in a year and a clear desert pixel
206.

**Token.** The network reads each observation the way a language model reads a word: as one
**token** in a sequence. A pixel with 180 optical and 120 radar observations is a
300-token sequence. **Tokens per pixel** is therefore just optical depth plus radar depth,
and it is the unit that matters for memory.

**Bucket.** A graphics card wants rectangular work — every sequence in one call the same
length. Pixels do not oblige, so the pipeline sorts them into **buckets** by depth. All the
pixels in one bucket are processed at the same sequence length; different buckets have
different lengths. The allowed lengths are a fixed list called the **checkpoint ladder**
(`num_obs_checkpoints`), which by default runs 8, 16, 32, … up to **256**. A pixel is
rounded up to the next rung, and — the load-bearing part — anything deeper than the top rung
is **clipped** to it.

**Sub-batch, and `batch_size`.** A bucket can hold millions of pixels, far more than a card
can hold at once, so it is processed in slices. One slice is a **sub-batch**, and
`batch_size` is how many pixels are in it. This is the number the change is about. Its
default is **7,168 pixels**.

One more piece of vocabulary, for the machinery rather than the model. An **actor** is one
worker process holding one copy of the network on one card; the pipeline runs hundreds of
them at once through the Ray framework. A **fill** is one unit of campaign work — one UTM
zone for one year. A **chunk** is a tile of pixels within a fill; a chunk that fails is
retried, up to three attempts, after which it is recorded `PERMANENTLY FAILED`.

### The memory rule, in words

When the card runs one sub-batch it must hold the intermediate results for every token in
it — the **activations**. There are `batch_size` pixels in the slice, each carrying up to
its bucket's tokens, so:

> **Memory needed ≈ (pixels in the slice) × (tokens each pixel carries).**

Both factors matter and only one of them is ours. `batch_size` is a setting. Tokens per
pixel is decided by the weather and the satellite's orbit over whatever piece of ground the
fill happens to cover. **That is the whole problem**: a batch size chosen once, on one card,
against one depth, is a bet that neither the card nor the depth will change. In August 2026
both did.

The network's own weights are not the issue: the loaded checkpoint accounts for about
**0.2 GiB** of the 20.9 GiB that was in use when the card ran out. Essentially all of the
memory is activations.

---

## 2. What went wrong, on 29 August 2026

The 2026-08-29 global campaign was the first to run two card types at once: the **L40S**
(44.4 GiB usable) in the `g6e.xlarge` instance, which had always been the only rung, and the
**A10G** (22.06 GiB usable) in `g5.2xlarge`, added by PR #159 as a capacity fallback for
when Amazon has no `g6e` to rent.

Within ten hours the fills had logged roughly **4,100 actor deaths**. The run immediately
before it — ten fills over 23 hours, L40S only — logged **zero**.

Every dead actor names its host in a `Replacing dead actor N (was on i-...)` line. Resolving
those instance ids against EC2 gives an unambiguous split:

| instance type | card | usable memory | dead-actor hosts |
|---|---|---:|---:|
| `g5.2xlarge` | A10G | 22.06 GiB | **178** |
| `g6e.xlarge` | L40S | 44.4 GiB | **0** |

That is 178 of 178. The Ray logs give the mechanism with no interpretation required:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.54 GiB.
GPU 0 has a total capacity of 22.06 GiB of which 1.17 GiB is free.
Including non-PyTorch memory, this process has 20.89 GiB memory in use.
```

The card was asked for 2.54 GiB it did not have. The process died, Ray replaced it, the
replacement re-loaded the 175 MB checkpoint and picked up the work — and, running the same
batch on the same card at the same depth, often died the same way.

---

## 3. The memory rule, measured

Section 1 asserted that memory is proportional to `batch_size × tokens per pixel`. That does
not have to be argued from the code. A forward sweep on a real A10G, recorded in
[`gpu-card-choice-2026_08.md`](gpu-card-choice-2026_08.md), measured it directly: the
production batch of 7,168 pixels, in bf16, at a range of sequence lengths, with radar depth
held at 0.4 × optical depth (the campaign's observed ratio).

`t_s2` is the optical sequence length in timesteps, `t_s1` the radar one; their sum is
tokens per pixel. **VRAM allocated** is bytes the allocator handed to live tensors.

| optical `t_s2` | radar `t_s1` | tokens per pixel | VRAM allocated (GiB) |
|---:|---:|---:|---:|
| 8 | 3 | 11 | 0.88 |
| 32 | 13 | 45 | 3.09 |
| 64 | 26 | 90 | 6.05 |
| 96 | 38 | 134 | 9.01 |
| 128 | 51 | 179 | 11.97 |
| 160 | 64 | 224 | 14.93 |
| 192 | 77 | 269 | 17.89 |
| 224 | 90 | 314 | 20.84 |

Plotted against tokens per pixel — not against optical depth alone — that is a straight
line:

> **allocated GiB = 0.0660 × (tokens per pixel) + 0.14**, at `batch_size = 7,168`

| statistic | value | meaning |
|---|---:|---|
| R² | 0.999992 | the line explains essentially all of the variation |
| largest residual | 0.03 GiB | worst disagreement between line and measurement |
| slope, per token | 9.65 KiB | memory cost of one token, at this batch |
| intercept | 0.14 GiB | the fixed part: weights, workspace, allocator overhead |

Two checks that this is a law and not a curve fit — neither was used to produce the line.

**Check 1, against the same document's refusal boundary.** A second, narrower sweep pushed
the A10G until it refused. Its last working rung was `t_s2 = 232`, which is 325 tokens per
pixel, and recorded 21.58 GiB. **The line predicts 21.59 GiB.** The next rung up, 336 tokens
per pixel, is predicted at 22.31 GiB against a 22.06 GiB card — and that is the rung that
refused.

**Check 2, against the production failure.** The traceback above needed 20.89 + 2.54 =
**23.43 GiB**. Inverting the line puts that at **353 tokens per pixel**. Independently: the
deepest optical depth ever recorded in this campaign is 206, which the ladder rounds to 208;
the deepest radar depth measured anywhere is 147, which rounds to 152; the sum is **360
tokens per pixel**, which the line predicts at 23.90 GiB. The line reproduces a production
out-of-memory event it was never fitted to, to within 2%.

### Why the deaths clustered in particular fills rather than spreading evenly

This looked at first like evidence against a simple memory rule. It is the opposite — it is
what the rule predicts.

At `batch_size = 7,168`, rearranging the line for a 22.06 GiB card gives a wall at **332
tokens per pixel**. And what has to fit is a fill's **deepest** bucket, not its average
pixel: the scheduler deliberately runs the deepest bucket first, so a chunk containing any
deep pixels pays its peak immediately.

| quantity | value |
|---|---:|
| campaign land-weighted mean | 173 tokens per pixel |
| optical depth, observed range | 57 – 206 |
| radar depth, observed range | 66 – 147 |
| correlation of optical depth with latitude band | Pearson r = +0.912 |
| correlation of radar depth with latitude band | Pearson r = +0.009 |
| where `B = 7,168` fills a 22.06 GiB card | 332 tokens per pixel |

The mean is comfortably under the wall. But optical depth tracks latitude strongly and radar
depth does not track it at all, so the two go deep **independently**, and the places where
both happen to go deep are scattered and geographically clustered. Those are the fills that
died. Nothing about their imagery was anomalous; they were simply on the far side of a
threshold the batch size had no way of knowing about.

---

## 4. The fix

`batch_size_for_gpu` computes the batch each actor should run, once, on the actor's own
device, before anything reads it. The calculation is a single ratio applied to the
**calibrated** batch of 7,168:

> **fitted batch = 7,168 × (this actor's share of the card ÷ 44 GiB) × (512 ÷ the ladder's
> deepest tokens per pixel)**, capped at whatever the caller asked for.

44 GiB and 512 tokens per pixel are what the calibration covered. The result on the two
production cards: **the L40S is untouched at 7,168; the A10G gets 3,593.**

### Why one number is enough — the worst case is bounded in advance

The reason a single fitted value can be safe for every bucket is the clipping mentioned in
section 1. A pixel's sequence is **not** open-ended: each of the two streams is clipped to
the ladder's deepest rung, 256, so **no bucket the sampler can build carries more than 512
tokens per pixel.** That is a property of the bucketing, fixed before any imagery is read,
and completely independent of geography.

So the deepest sub-batch that can ever be presented to a card is `batch × 512` tokens, and
that is what the batch has to survive:

| configuration | batch | deepest legal sub-batch | predicted VRAM | of the card |
|---|---:|---:|---:|---:|
| L40S, unchanged | 7,168 | 3,670,016 tokens | 33.9 GiB | 76% of 44.4 GiB |
| A10G, before this change | 7,168 | 3,670,016 tokens | 33.9 GiB | **154% of 22.06 GiB** |
| **A10G, fitted** | **3,593** | **1,839,616 tokens** | **17.1 GiB** | **77% of 22.06 GiB** |

Two independent ways of reading the A10G row:

* **Against measurement.** 1,839,616 tokens is **79% of the 2,329,600 an A10G was measured
  to complete** (7,168 pixels at the 325-token rung, the deepest that worked in the narrow
  sweep).
* **Against the deepest depth ever actually observed.** At 360 tokens per pixel the fitted
  batch needs 12.1 GiB — **55% of the card**.

The table also shows why the L40S needs no change and gets none: at 76% of its memory it
cannot reach an out-of-memory condition on any bucket the sampler can legally build. **That
is a falsifiable prediction.** If any chunk recorded `PERMANENTLY FAILED` on 2026-08-29
turns out to have died on a `g6e.xlarge`, this line is wrong and the derivation must be
re-opened.

Note that scaling by memory holds the deepest legal sub-batch at nearly the same fraction of
every card — 76% on the L40S, 77% on the A10G. That is not luck. It is what a linear memory
law and a memory-proportional batch necessarily produce together, and it is why one ratio
generalises to cards nobody has measured yet.

---

## 5. Why this approach and not the others

Four alternatives were on the table. Each is stated as its advocate would state it.

### Alternative A — give each bucket its own budget, from its actual depth

**The proposal.** Stop shipping a pixel count at all. Ship a *token* budget, and at each
bucket compute `batch = budget ÷ that bucket's tokens per pixel`. Deep buckets automatically
get small slices; shallow buckets get large ones. It is the exact expression of the memory
rule rather than a conservative approximation of it, and it is the first idea anyone has on
learning that memory depends on depth.

**Why it was rejected — two measurements.**

*It solves a problem that is already solved.* The clipping bound above means the worst case
is known before a tile is read, and one fitted value clears it with 21–23% margin against
measurement. A per-bucket rule would be adapting to a quantity that is already capped.

*Its only remaining benefit is throughput, and the throughput is not there.* What a
per-bucket budget could still buy is the full 7,168 pixels back on shallow buckets, where
the fitted batch is conservative. But adaptive token-budget batching was measured on the
real model and shelved — see the dead-end list in
[`inference_gpu_saturation_profile_2026_07.md`](inference_gpu_saturation_profile_2026_07.md):
**`B = 7,168` is throughput-optimal at *every* sequence length**, down to 8 timesteps, and
larger batches are neutral-to-worse. The card is already saturated at these sizes: on the
A10G, **SMACT 0.995** — the fraction of time its arithmetic units are busy — at its full
1710 MHz clock. There is no idle capacity for a bigger shallow-bucket batch to fill.

**What it would cost.** A second calibration constant, a per-bucket branch in the hot loop,
and a new failure mode: a mis-set token budget becomes a per-bucket out-of-memory rather
than a whole-run one, which is harder to see and harder to reproduce. Paid on a fallback
rung the economics say we would rather not be using at all.

**How to reopen it.** If a future measurement shows throughput rising with batch size at
shallow depth on a small card, this reasoning no longer holds and the budget is worth
building.

### Alternative B — just halve the number for the A10G and move on

**The proposal.** One line: if the card is an A10G, use 3,584. No arithmetic, no constants,
no ratio, and it would have stopped the 29 August failures.

**Why it was rejected.**

*It only knows the cards you happened to enumerate.* `fleet_mix.GPU_RUNGS` decides which
instance rungs the campaign may open, and it is a list that changes. A 24 GiB L4, a 40 GiB
A100, an 80 GiB H100 — each gets either a wrong answer or no answer, and the failure mode
for "no answer" is the 7,168 default, which is the bug.

*It is not a statement about anything, so it cannot be checked.* The ratio version makes a
claim — *hold the deepest legal slice at a fixed fraction of whatever card is present* —
that the linear memory law makes true for cards that have never been benchmarked, and that a
test can pin. A hardcoded pair of numbers can only be checked against the two cards it was
written for.

*It has nowhere to put the other two inputs.* Automated review found two further ways to
exceed the cap (section 6). Both fit naturally into a ratio and not at all into a card-name
lookup.

### Alternative C — use the smaller batch everywhere

**The proposal.** Set the default to 3,593 for all cards. No branching, no per-device
computation, one number in the config, and obviously safe on both.

**Why it was rejected.** It pays a permanent throughput tax on the card that does nearly all
of the work, to protect a card that is only rented when the primary one is unavailable. The
cost is measured rather than assumed: the pipeline's own baseline ran at `batch_size = 3,584`
before the July 2026 optimisation campaign, and the per-sub-batch CPU preparation cost is
roughly **flat in batch size** — 165 ms for a 3,584-pixel sub-batch, against 103–160 ms for
a 7,168-pixel one. Halving the batch therefore roughly doubles that fixed cost per pixel,
and doubles the number of forward launches for the same imagery. (The full 3,584-to-7,168
comparison in the saturation profile shows 2–2.8× per-worker throughput, but that branch
carried several other optimisations, so the batch cannot claim all of it. The per-sub-batch
preparation cost can be attributed cleanly, and it is enough to settle the question.)

### Alternative D — refuse the small card

**The proposal.** Close the `g5.2xlarge` rung. The batch is correct for the L40S; run only
on L40S.

**Why it was rejected.** The rung exists because of capacity, not preference. PR #159 opened
`g5.2xlarge` precisely because Amazon could not supply `g6e.xlarge` at fleet width, and
refusing it means the campaign stalls rather than running somewhat slower on a smaller card.
The fallback is worth having; it just needed a batch size that fits it.

---

## 6. What the fit has to see, and why

"The batch needs no per-bucket term" is not the same claim as "memory is the only input".
The 512-token cap in section 4 is a *consequence* of two configured things, and a fit that
ignored them would be safe only by coincidence. Automated review found both, plus a third
route around the calculation, and they compound rather than compete.

**A deeper ladder.** `num_obs_checkpoints` is an ordinary config field and the pipeline
accepts any positive depths. A caller passing `(512,)` doubles every sub-batch's token count.
A memory-only ratio would still hand the A10G 3,593 pixels — a 3,679,232-token sub-batch,
158% of the measured ceiling.

**A packed card.** A fractional Ray reservation (`num_gpus`) deliberately puts several actors
on one card. Each of them sizing to the card's *total* memory oversubscribes it by exactly
the packing factor. The share an actor actually receives is `1 ÷ floor(1 ÷ num_gpus)`,
because that is how many actors Ray fits — so `num_gpus = 0.6` packs one actor that owns the
*whole* card, while `0.4` packs two that get half each. Reading the reservation itself as the
share is wrong in both directions: it needlessly cuts the 0.6 actor's batch, and it
under-counts the memory pressure at 0.4.

**A caller asking for more than was calibrated.** `batch_size` is a settable field. Scaling
*the caller's request* by the card ratio preserves an over-ask: a request of 10,000 on an
A10G came out at 5,013, a 2,566,656-token sub-batch, 110% of the measured ceiling. The fit is
therefore derived from the calibrated 7,168 and the caller's number applied only as a
ceiling.

All three are handled the same way — one scale factor, still computed once in the actor —
and the demand placed on a card is then invariant across configurations that ought to be
equivalent:

| ladder | reservation | actors on the card | fitted batch | tokens demanded of the card | of measured ceiling |
|---|---|---:|---:|---:|---:|
| default (deepest 256) | whole card | 1 | 3,593 | 1,839,616 | 79% |
| default (deepest 256) | 0.5 | 2 | 1,796 | 1,839,104 | 79% |
| `(512,)` | whole card | 1 | 1,796 | 1,839,104 | 79% |
| `(512,)` | 0.5 | 2 | 898 | 1,839,104 | 79% |
| default (deepest 256) | 0.1 | 10 | 359 | 1,838,080 | 79% |
| `(4096,)` | whole card | 1 | 224 | 1,835,008 | 79% |

Same card, same demand, six configurations. Before the fit took all three inputs, the last
two rows read **2,621,440 tokens (113%)** and **4,194,304 tokens (180%)** respectively. The
L40S at the default ladder on a whole card still gets 7,168, unchanged, because its share
already exceeds the calibration reference.

### Two deliberate choices inside that

**There is no minimum batch size.** An earlier version floored the fitted value at 512
pixels, on the reasoning that below that the per-forward overhead dominates and a small card
is starved rather than protected. That floor is what produced the 113% and 180% rows above:
it silently raised the batch back over the bound in exactly the configurations nobody
watches. A configuration whose safe batch is 224 pixels now gets 224 pixels and runs slowly.
Slow is a cost; out-of-memory is a failure.

**A configuration is never refused.** Review's alternative suggestion was to reject
configurations whose safe batch falls below an operational floor. Refusing turns a slow run
into a dead one, and the configurations in question — a 0.1 GPU reservation, a 4,096-deep
ladder — are not production settings and have no operator waiting to fix them. Running
correctly and slowly is the better failure.

**One caveat, stated because the table above does not show it.** The invariant is on
*activations*. Each actor also pays fixed per-process costs — its own CUDA context and its
own copy of the weights — which the token accounting does not model. At the default of one
actor per card that is 0.14 GiB and irrelevant. At ten actors per card it is ten times that,
and the true demand is nearer 83% of the card than 79%. Heavy packing is not a production
configuration, and this is one of the reasons.

---

## 7. What the margin rests on

Two assumptions. Both now fail a test rather than fail a fill.

**The token cap itself.** `test_the_fitted_batch_holds_the_deepest_bucket_the_sampler_can_build`
pins `fitted × 2 × max(checkpoints) × actors-per-card` against the measured ceiling, over
every combination of ladder depth and reservation in the table above.
`test_the_unfitted_batch_does_not_hold_it` proves that bound is not vacuous by showing the
old batch fails the same test. Dropping either the depth term or the packing term from the
fit turns exactly the affected rows red; this was verified by doing it.

**Segment-backed allocation.** Every VRAM figure in this document is *allocated* bytes —
memory handed to live tensors. Under PyTorch's default caching allocator the *reserved* pool
runs well above that and strands the difference: **20.58 GiB reserved for an 11.97 GiB
working set**, measured. That is why the A10G's refusal boundary sits at optical depth 208
without the flag and 232 with it. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ships on
the actor's Ray `runtime_env` (merged in #154) and is now pinned by
`test_the_actor_ships_the_segment_backed_allocator`. Removing it would re-open this failure
silently, on the fallback rung only.

---

## 8. What it cost while unfixed

**In the first ten fills, nothing permanent.** A chunk gets three attempts before it is
recorded `PERMANENTLY FAILED`, and a scan of every ERROR record across all ten found zero.
The cost there was GPU time: roughly 4,100 actor restarts in ten hours, each one re-loading a
175 MB checkpoint before the replacement could take work. The **7,830 chunks/hour** measured
that afternoon is therefore a figure *including* this churn, not a clean throughput number.

**Then it stopped being free.** Between 16:27Z and 16:54Z on 2026-08-29 a single fill
recorded **16 chunks `PERMANENTLY FAILED`**, every one an out-of-memory on the third and
final attempt. Three attempts at the same chunk, on the same card, at the same batch size, is
not three chances at all: the sub-batch either fits or it does not, and the entire retry
budget was being spent on a deterministic refusal.

---

## 9. Note for whoever merges this

It changes the `inference` package, so it moves the **staging fingerprint** — the identity
the pipeline computes over its own source to decide whether previously staged inputs can be
reused. A campaign resuming previously staged tiles must be dispatched with the existing
`staging_code_identity`, exactly as the 2026-08-29 restart was. See
[`staging-identity-and-resume.md`](staging-identity-and-resume.md).

---

## Appendix: where each thing lives in the code

| what | where |
|---|---|
| the batch calculation | `config/inference.py`, `batch_size_for_gpu` |
| its constants (44 GiB, 7,168 pixels, 512 tokens) | `config/inference.py`, `TUNED_GPU_GIB`, `TUNED_BATCH_SIZE`, `TUNED_TOKENS_PER_PIXEL` |
| where an actor applies it | `inference/actors.py`, `InferenceActor.__init__` |
| reading the card's size | `inference/actors.py`, `_gpu_total_gib` |
| the checkpoint ladder and its clipping | `inference/sampling.py`, `compute_bin_keys` |
| deepest bucket first | `inference/dataset.py`, `iter_buckets(largest_first=True)` |
| how many actors Ray packs on a card | `inference/scheduling.py`, `FleetDemand.machines` |
| which instance rungs may be opened | `providers/aws/fleet_mix.py`, `GPU_RUNGS` |
| the allocator flag | `inference/actors.py`, the `@ray.remote(runtime_env=...)` decorator on `InferenceActor` |

`batch_size` is read in two different places downstream, both in `inference/inference.py` —
the sub-batch split in `run_inference`, and the pinned host buffers allocated by
`_pipelined_gpu_loop`. That is why the actor
narrows it **once**, in its constructor, and stores the result on `self.config`: scaling at
either read site would leave the other on the tuned value, and the two would disagree only
on the card where it matters. `test_the_actor_reads_no_un_narrowed_config_after_fitting_the_batch`
pins that structurally.
