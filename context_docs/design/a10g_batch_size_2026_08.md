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
|---|---|
| `g5.2xlarge` (A10G) | **178** |
| `g6e.xlarge` (L40S) | **0** |

That is 178 of 178. The mechanism is in the Ray actor logs, unambiguous:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.54 GiB.
GPU 0 has a total capacity of 22.06 GiB of which 1.17 GiB is free.
Including non-PyTorch memory, this process has 20.89 GiB memory in use.
```

## Why the batch is the lever

The checkpoint is ~0.2 GiB. The 20.9 GiB in use is therefore almost entirely activations,
and activations scale with `batch_size` — the tuned 7,168 pixels per sub-batch. The two
cards differ by almost exactly a factor of two in reported memory:

| card | instance | reported total | nominal |
|---|---|---|---|
| L40S | `g6e.xlarge` | 44.7 GiB | 48 GB |
| A10G | `g5.2xlarge` | 22.06 GiB | 24 GB |

Scaling the batch by that ratio gives the A10G 3,593 and leaves the L40S untouched. The
projected requirement falls from 23.4 GiB — which is what it was asking for against a
22.06 GiB card — to roughly 11.7 GiB, so there is real headroom rather than a value that
merely fits today's chunk shapes.

## What it cost while unfixed

Nothing permanent. A chunk gets three attempts before it is recorded `PERMANENTLY FAILED`,
and a scan of every ERROR record across all ten fills of that run found **zero** permanent
failures — the retries absorbed it. The cost was GPU time: ~4,100 actor restarts in ten
hours, each reloading a 175 MB checkpoint before the replacement could take work. The
7,830 chunks/hour the campaign measured that afternoon is therefore a figure *including*
this churn, and the fix should raise it rather than merely stabilise it.

## Why the policy keys on memory rather than on the instance type

The same reason PR #159's rung marker is ours rather than Ray's autofilled
`accelerator_type:<CARD>`: the causal quantity is how much memory the card has, and a
policy written against that needs no edit when the next rung is added. `fleet_mix.GPU_RUNGS`
is designed to grow, and a table of instance types here would have to grow with it.

## Note for whoever merges this

It changes the `inference` package, so it moves the staging fingerprint. A campaign
resuming previously staged tiles must be dispatched with the existing
`staging_code_identity`, exactly as the 2026-08-29 restart was — see
`staging-identity-and-resume.md`.
