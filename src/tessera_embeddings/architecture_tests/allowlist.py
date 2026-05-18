"""TOML allowlist parsing for downstream-extensible architecture rules.

The OSS bundled rules ship strict allowed-path lists. Downstream
consumers (closed-source repos, community-adapter PRs) sometimes need
to permit a forbidden import in additional paths — for example, a
closed-source ``yield_modeling`` repo may legitimately import
``boto3`` from its own ``yield_modeling/iac/`` Pulumi stack code.

The allowlist TOML lets them extend the rules without forking::

    # downstream-arch-allowlist.toml
    [allowed_imports."no-boto3-outside-aws-provider"]
    paths = ["yield_modeling/providers/aws/", "yield_modeling/iac/"]

    [allowed_imports."no-prefect-outside-prefect-layer"]
    paths = ["yield_modeling/orchestration/prefect/"]
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def load_allowlist(path: Path) -> dict[str, tuple[str, ...]]:
    """Parse a TOML allowlist file into ``{rule_name: extra_paths}``.

    Args:
        path: Path to the TOML file.

    Returns:
        Dict suitable for the ``extra_allowed_paths`` parameter of
        :func:`tessera_embeddings.architecture_tests.run`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is malformed.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)

    section = data.get("allowed_imports", {})
    if not isinstance(section, dict):
        raise ValueError(f"Expected [allowed_imports] table in {path}, got {type(section).__name__}")

    out: dict[str, tuple[str, ...]] = {}
    for rule_name, entry in section.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected [allowed_imports.{rule_name}] to be a table, got {type(entry).__name__}"
            )
        paths = entry.get("paths", [])
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError(f"[allowed_imports.{rule_name}].paths must be a list of strings")
        out[rule_name] = tuple(paths)
    return out
