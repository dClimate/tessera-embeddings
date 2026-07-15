#!/usr/bin/env -S uv run python
"""Build / verify / validate the campaign land-mask coverage store (ADR-010).

Turns the partner's per-0.1°-cell TIFF delivery
(``s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/``) into per-zone
tile-liveness bitmaps in one Icechunk repo. Pure geometry over ``registry.txt``
— no delivery pixels are read except by ``verify``'s spot-check. Runs on a
laptop in well under a minute (a couple of minutes with a full spot-check).

Subcommands::

    # Guard the v1.1 all-1s assumption before trusting a build:
    ./scripts/build_landmask_coverage.py verify --spot-check 500

    # Build all 120 zone groups into a destination repo (one commit):
    ./scripts/build_landmask_coverage.py build --dest s3://.../masks/global.icechunk

    # Structural + geographic self-checks on a built store:
    ./scripts/build_landmask_coverage.py validate --dest s3://.../masks/global.icechunk

Credentials use the ambient AWS chain (env / instance profile). ``--zones`` (a
comma-separated EPSG list) restricts build/validate for a quick local run.
"""

from __future__ import annotations

import argparse
import logging
import sys

from tessera_embeddings.ingest.land_mask import (
    DEFAULT_DELIVERY_URI,
    REGISTRY_NAME,
    build_all,
    read_registry,
    reconcile_with_bucket,
    spot_check_delivery,
    validate_coverage,
)

log = logging.getLogger("build_landmask_coverage")


def _registry_uri(args: argparse.Namespace) -> str:
    return args.registry or f"{args.delivery.rstrip('/')}/{REGISTRY_NAME}"


def _zones(args: argparse.Namespace) -> list[str] | None:
    return [z.strip() for z in args.zones.split(",") if z.strip()] if args.zones else None


def cmd_verify(args: argparse.Namespace) -> None:
    """Reconcile the registry against the bucket and spot-check delivery tiles."""
    names, sha = read_registry(_registry_uri(args))
    log.info("Registry: %d cells, sha256 %s", len(names), sha)
    if not args.no_reconcile:
        reconcile_with_bucket(names, delivery_uri=args.delivery, log=log)
    if args.spot_check > 0:
        spot_check_delivery(names, delivery_uri=args.delivery, n=args.spot_check, log=log)


def cmd_build(args: argparse.Namespace) -> None:
    """Build per-zone coverage bitmaps into ``--dest`` (one commit)."""
    result = build_all(
        args.dest,
        registry_uri=_registry_uri(args),
        delivery_uri=args.delivery,
        zones=_zones(args),
        log=log,
    )
    log.info("Built %s", result)


def cmd_validate(args: argparse.Namespace) -> None:
    """Run structural + geographic self-checks on a built coverage store."""
    validate_coverage(args.dest, zones=_zones(args), log=log)
    log.info("Coverage store %s validated OK", args.dest)


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", default=DEFAULT_DELIVERY_URI, help="Partner delivery prefix.")
    parser.add_argument("--registry", default=None, help="Registry URI (default: {delivery}/registry.txt).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Reconcile registry vs bucket + spot-check tiles.")
    p_verify.add_argument("--spot-check", type=int, default=500, help="TIFFs to sample (0 = skip).")
    p_verify.add_argument("--no-reconcile", action="store_true", help="Skip the (slow) full bucket listing.")
    p_verify.set_defaults(func=cmd_verify)

    p_build = sub.add_parser("build", help="Build per-zone coverage bitmaps into --dest.")
    p_build.add_argument("--dest", required=True, help="Coverage Icechunk repo URI.")
    p_build.add_argument("--zones", default=None, help="Comma-separated EPSG list (default: all 120).")
    p_build.set_defaults(func=cmd_build)

    p_validate = sub.add_parser("validate", help="Self-check a built coverage store.")
    p_validate.add_argument("--dest", required=True, help="Coverage Icechunk repo URI.")
    p_validate.add_argument("--zones", default=None, help="Comma-separated EPSG list (default: all 120).")
    p_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
