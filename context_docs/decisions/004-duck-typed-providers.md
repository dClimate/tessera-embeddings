# 004 — Duck-typed providers, not abstract `Provider` classes

**Status:** Accepted (v0.1.0)

## Context

Concrete cloud-provisioning glue lives under `providers/<cloud>/`.
We needed to decide how the orchestration layer addresses
providers: through an abstract interface, or by direct import.

* **Option A: abstract `Provider` base class.** Each cloud
  subclasses; flows accept `provider: Provider`.
* **Option B: registry / entry-point plugin system.** Providers
  register themselves; flows look them up by name.
* **Option C: duck-typed context managers.** Each provider is a
  module exposing `ray_cluster(...)` and `dask_cluster(...)`
  context managers. Flows import the one they want.

## Decision

**Option C.** Each provider is a sibling directory under
`providers/`. Flows import their provider explicitly:

```python
from tessera_embeddings.providers.aws.dask import ecs_cluster
from tessera_embeddings.providers.aws.ray import ray_cluster
```

To swap providers, edit the import or fork the flow file. There's
no `Provider.create("aws")`-style registry, no entry points.

## Rejected alternatives

**Option A (abstract base class):** Cloud APIs differ at the kwarg
level. AWS's `ray_cluster` needs `ami_ssm_name`; GCP's would need
`image_uri`. An abstraction that fits both is either too thin to
help (everything is `**kwargs`) or too thick to evolve (forces
every new cloud to invent a way to express its quirks within the
shared schema). Tested with prototypes; both extremes were ugly.

**Option B (registry):** Adds runtime indirection without solving
the kwarg problem. Worse, it lets misconfiguration produce
"provider not found" runtime errors instead of an `ImportError` at
load time. We prefer fast, loud failures.

## Consequences

- **Pro:** providers are independent. Adding GCP doesn't change
  AWS's code.
- **Pro:** "what does the AWS provider do?" is answered by
  reading `providers/aws/ray.py` top-to-bottom. No abstraction
  layer to trace through.
- **Pro:** community-contributed providers can ship in their own
  `providers/<their_cloud>/` without coordinating with core.
- **Con:** there's no programmatic way to enumerate providers. If
  a downstream needs a provider registry, it'll write one over our
  modules.
- **Con:** "swap providers" requires either a code edit or a
  user-side conditional (`if cloud == "gcp": ... else: ...`).
  Acceptable — the alternative was an interface that nobody could
  use without escape hatches.

## Related

- [`docs/providers/adding-your-own.md`](../../docs/providers/adding-your-own.md) —
  the worked example that follows from this decision.
- [`src/tessera_embeddings/providers/README.md`](../../src/tessera_embeddings/providers/README.md) —
  user-facing summary.
