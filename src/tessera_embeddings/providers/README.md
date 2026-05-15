# providers/

Cloud-substrate provisioning for the reference orchestration layer.

Each provider owns a concrete cloud (or local) implementation of two
substrates: Ray clusters (for inference) and Dask clusters (for ingest
and assembly). There is no abstract `Provider` base class — providers
are duck-typed context managers that yield substrate handles.

## Layout

```
providers/
├── aws/      # AWS reference: ECS Fargate Dask, EC2/Ray cluster launcher
└── local/    # Local-machine fallbacks for tests, demos, plain.py
```

Adding a new provider (GCP, Azure, k8s, on-prem) means writing a new
sibling directory with the same interface — see `docs/providers/adding-your-own.md`
in Phase 12.
