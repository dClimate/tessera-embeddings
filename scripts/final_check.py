"""Verify every figure the branch now publishes against the locally-computed values.

Fails loudly on any figure that is not present where it should be, or that appears where it was
withdrawn. Point of this is that "I swept it" is a claim to be tested, not asserted.
"""
import pathlib, re, sys

FILES = {
    "doc": "context_docs/assembly/what-bounds-assembly-2026-09-09.md",
    "reg": "context_docs/corrections-register.md",
    "cfg": "docs/configuration.md",
    "pfx": "docs/prefect-setup.md",
    "code": "src/tessera_embeddings/config/assembly.py",
    "store": "context_docs/storage/writing-to-the-global-store.md",
    "cost": "context_docs/campaign/campaign-cost-model.md",
    "idx": "context_docs/README.md",
    "t1": "tests/unit/config/test_config_assembly.py",
    "t2": "tests/unit/orchestration/flows/test_run_global_campaign.py",
}
text = {k: pathlib.Path(v).read_text() for k, v in FILES.items()}
fail = []

# MUST be present: the confirmed, attributable figures.
must = [
    ("doc", "40.1 GiB", "16-worker per-task memory peak"),
    ("doc", "93.5 GiB", "32-worker per-task memory peak"),
    ("doc", "62.7%", "40.1 of 64 GiB"),
    ("doc", "38.3%", "93.5 of 244 GiB"),
    ("doc", "95.8%", "16-worker per-task CPU peak"),
    ("doc", "99.7%", "32-worker per-task CPU peak"),
    ("doc", "2.51 GiB", "16-worker ratio"),
    ("doc", "2.92 GiB", "32-worker ratio"),
    ("doc", "FARGATE", "launch type recorded"),
    ("doc", "10,000-row cap", "truncation checked"),
    ("cfg", "near 40 GiB", "public guidance uses the measured peak"),
    ("cfg", "full 64 GiB", "public guidance sizes the runner"),
    ("pfx", "near 40 GiB", "second public doc"),
    ("code", "near 40 GiB", "docstring"),
    ("code", "near 94 GiB", "docstring 32-worker"),
    ("store", "logical", "read-rate relabelled"),
    ("cost", "logical", "cost model relabelled"),
]
for key, needle, why in must:
    if needle not in text[key]:
        fail.append(f"MISSING in {FILES[key]}: {needle!r} ({why})")

# MUST NOT be present: withdrawn figures, outside a withdrawal passage.
banned = [
    ("cfg", "24 GB", "the old under-provisioning guidance"),
    ("pfx", "24 GB", "the old under-provisioning guidance"),
    ("pfx", "1–1.5 GB", "the old per-worker estimate"),
    ("code", "1-1.5 GB", "the old per-worker estimate"),
    ("cfg", "20 GB measured peak", "the old false peak"),
    ("pfx", "20 GB measured peak", "the old false peak"),
    ("doc", "87.8", "a figure that was itself wrong"),
    ("cfg", "linear in worker count", "the withdrawn linearity claim"),
    ("code", "linear in worker count", "the withdrawn linearity claim"),
    ("t1", "2.9 GiB", "withdrawn coefficient in a test docstring"),
    ("t2", "2.9 GiB", "withdrawn coefficient in a test docstring"),
    ("t1", "47 GiB", "withdrawn figure"),
    ("store", "effective read rate", "mislabelled rate"),
    ("cost", "effective read rate", "mislabelled rate"),
]
for key, needle, why in banned:
    if needle in text[key]:
        fail.append(f"STILL PRESENT in {FILES[key]}: {needle!r} ({why})")

# Arithmetic the document asserts.
checks = [
    ("40.1/64", 40.1 / 64 * 100, 62.7),
    ("93.5/244", 93.5 / 244 * 100, 38.3),
    ("46.7/64", 46.7 / 64 * 100, 73.0),
    ("40.1/16", 40.1 / 16, 2.51),
    ("93.5/32", 93.5 / 32, 2.92),
    ("slope", (93.5 - 40.1) / 16, 3.34),
    ("intercept", 40.1 - 16 * (93.5 - 40.1) / 16, -13.3),
    ("15693/16384", 15693.5 / 16384 * 100, 95.8),
    ("32654/32768", 32654.7 / 32768 * 100, 99.7),
]
for name, got, want in checks:
    if abs(got - want) > 0.06:
        fail.append(f"ARITHMETIC {name}: computed {got:.3f}, document says {want}")

if fail:
    print("FAILURES:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print(f"all {len(must)} required figures present, {len(banned)} withdrawn figures absent, "
      f"{len(checks)} arithmetic checks agree")
