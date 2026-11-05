# Adapter parity template

Community-contributed orchestrator adapters (Dagster, Airflow, Argo,
Kubeflow, …) must include a parity test before merge. This directory
holds the starter template: copy it to a sibling directory named after
your adapter and add the test below.

## Steps

1. Copy this directory::

       cp -r tests/parity/adapter_template tests/parity/<adapter_name>

2. Implement your adapter under
   ``src/tessera_embeddings/orchestration/<adapter_name>/`` with the
   same domain-function-delegation pattern as
   ``orchestration/prefect/``.

3. Add ``tests/parity/<adapter_name>/test_<adapter_name>_parity.py``
   from the template below, replacing ``run_via_your_adapter`` with
   your adapter's flow / pipeline invocation. It compares against the
   same domain function the bundled tests do.

4. Run::

       uv run pytest tests/parity/<adapter_name>/ -m parity

5. PR review checklist:
   - [ ] Architecture contracts pass (``uv run lint-imports`` and
     ``uv run ruff check src/`` both return 0).
   - [ ] Parity test passes locally.
   - [ ] CI workflow updated to include the new test directory.

## The test

The contract is identical for every adapter: run the adapter against the
same inputs as the domain function, then assert the output Zarr stores
are equivalent (modulo timestamps and run IDs — see
`tests.parity.helpers.assert_zarr_equivalent`).

```python
from pathlib import Path

import pytest

from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tessera_embeddings.orchestration.<your_adapter>.flows import run_via_your_adapter
from tests.parity.helpers import assert_zarr_equivalent


@pytest.mark.parity
def test_your_adapter_parity(tmp_path: Path, fixture_quickstart_roi: Path) -> None:
    kwargs = {"resolution": 10.0, "chunk_size": 2000, "force_crs": "EPSG:32613"}
    rasterize_roi_zarr(output_path=str(tmp_path / "domain.zarr"), input_path=str(fixture_quickstart_roi), **kwargs)
    run_via_your_adapter(output_path=str(tmp_path / "adapter.zarr"), input_path=str(fixture_quickstart_roi), **kwargs)
    assert_zarr_equivalent(tmp_path / "domain.zarr", tmp_path / "adapter.zarr")
```
