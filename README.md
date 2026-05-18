# tessera_embeddings

[![Lint](https://github.com/OWNER/tessera-embeddings/actions/workflows/lint.yml/badge.svg)](https://github.com/OWNER/tessera-embeddings/actions/workflows/lint.yml)
[![Unit tests](https://github.com/OWNER/tessera-embeddings/actions/workflows/unit.yml/badge.svg)](https://github.com/OWNER/tessera-embeddings/actions/workflows/unit.yml)
[![Architecture](https://github.com/OWNER/tessera-embeddings/actions/workflows/architecture.yml/badge.svg)](https://github.com/OWNER/tessera-embeddings/actions/workflows/architecture.yml)
[![Nightly](https://github.com/OWNER/tessera-embeddings/actions/workflows/nightly.yml/badge.svg)](https://github.com/OWNER/tessera-embeddings/actions/workflows/nightly.yml)

Distributed satellite data ingestion and Tessera embedding generation.

> Phase 1 scaffold. The full README ships in Phase 12 (see implementation plan).
> Replace `OWNER` in the badge URLs above with the actual GitHub org or user
> at first push.

## Status

Pre-release. APIs may change. See `context_docs/decisions/` for design records.

## Quickstart

```bash
uv sync --frozen
uv run python scripts/check_env.py
```

## License

Apache-2.0. See `LICENSE`.
