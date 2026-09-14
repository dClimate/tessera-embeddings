# 024 — `spatial:transform`'s origin is the pixel corner, and the published store is repaired in place

**Status:** Accepted 2026-09-14. Built on `fix/spatial-transform-corner-origin`.

Reported by a consumer of the published store: *"spatial:transform on your store seems to point to
the first pixel centre rather than the outer corner; I had to correct for that."* The report is
correct. This records what was wrong, what the evidence is, what changed, and the three things that
were deliberately not done.

---

## 1. The defect

Every one of the published store's 120 zone groups carried a `spatial:transform` whose origin was
the **centre** of the first pixel. The `spatial:` convention places it at the **outer corner**:

> `c`: Western-most coordinate of the X axis … `f`: Northern-most coordinate of the Y axis
>
> The transform operates on array indices where `(0, 0)` is at the **top-left corner** of the
> top-left pixel, and `(width, height)` is at the bottom-right corner of the bottom-right pixel.
> The center of the top-left pixel is at `(0.5, 0.5)`. This follows the GDAL geotransform and
> Python's Affine library convention.

The store also declares `spatial:registration: "pixel"`, under which the spec says the coordinate
ranges — naming `spatial:bbox` *and* `spatial:transform` — "refer to the outside edges of the
boundaries of the grid". Both readings give the same answer, and the wording is identical in the
tagged `v0.1` release and on `main`, so there is no version in which the shipped value was right.

The error was a uniform half pixel, 5 m, on every zone:

```
  zone 33N, as published          zone 33N, as it should be
  ─────────────────────           ─────────────────────────
  c = 163845.0    (centre)        c = 163840.0    (western edge)
  f = 9338875.0   (centre)        f = 9338880.0   (northern edge)

  a consumer trusting c, f placed every pixel 5 m EAST and 5 m SOUTH of its true
  position — half a cell diagonally, no rotation and no scale error
```

**The store contradicted itself, which is how this was confirmed rather than argued.**
`spatial:bbox` was already edge-based and exactly right on all 120 groups, spanning precisely
`shape × resolution` from the true corner. So the two attributes disagreed by half a pixel, and
either one could be checked against the other without appealing to the spec at all.

## 2. Where it came from

`storage/conventions.py` holds two helpers twenty lines apart. `_compute_bbox` adds half a pixel
back on all four sides, with a comment saying why. `_compute_affine_transform` took `x_coords[0]`
and `y_coords[0]` as the origin and never added it back. The coordinate arrays are pixel centres on
both paths that reach the builder — `zone_grid.easting_coords` constructs them as
`edge + (i + 0.5) × pitch`, and the single-ROI path reads them from a mosaic where `odc` wrote
centres (verified: an `odc.geo.GeoBox` built on a corner `Affine` returns `x[0] = corner + res/2`).
Neither caller was passing edges, so there was no code path in which the old value was correct.

**No test related the two helpers.** The only assertion on the transform pinned its literal value,
and the wrong value was what got pinned — a test can only defend the invariant somebody thought to
write down.

## 3. What changed

| what | where |
|---|---|
| the origin steps half a pixel back along each axis's own **signed** resolution | `storage/conventions.py`, `_compute_affine_transform` |
| an invariant test: `c == bbox[0]`, `f == bbox[3]`, and `shape × resolution` lands on the far edges, over three resolutions and a south-up grid | `tests/unit/storage/test_conventions.py`, `TestTransformAndBboxAgree` |
| the published store repaired in place, 120 groups, one commit | `scripts/maintenance/fix_published_store_spatial_transform.py` |
| the repair script exercised against a real Icechunk store regressed to the broken shape | `tests/unit/storage/test_fix_published_store_spatial_transform.py` |
| consumer-facing georeferencing section | `context_docs/storage/reading-the-published-store.md` §2a |

**Signed, not absolute.** A north-up grid has a negative `e`, so subtracting `e/2` moves the Y
origin *up* to the northern edge; an ascending Y axis moves *down* to its own leading edge. Using
`abs()` would land half a pixel inside the data on one of the two.

### The dead registration URLs, fixed in the same change

Separately found while reading the same attributes: the `zarr_conventions` entries for `proj:` and
`spatial:` pinned `refs/tags/v1`, a tag neither convention has cut, so all four URLs a consumer
might follow returned 404. The code had already hit this for `geoemb:` and worked around it by
pinning a commit, but left the other two. Both repositories carry `v0.1`; that is what they now
pin, and a test asserts it. `proj:` also moved organisation — `zarr-experimental/geo-proj` still
redirects, but `zarr-conventions` is the home.

## 4. Why the store is repaired rather than reissued or documented

**Repaired.** This is metadata on 120 group objects: one Icechunk commit, no data rewritten, and
the previous snapshot stays on `main`'s history so a rollback is `reset_branch`. Reissuing the
store to correct 5 m of metadata would cost the whole campaign. Documenting the deviation instead
would ask every consumer, forever, to apply a correction that the convention says they should not
have to — which is exactly what the reporter had to do.

**How the script earns the write.** It is the one artifact nobody can rebuild, so the script:

1. derives each group's corrected origin from that group's **own** coordinate arrays, never a table;
2. cross-checks the result against that group's **own** `spatial:bbox` — written at seeding by
   different code — and refuses unless the two agree independently. Without this the script would
   only be asserting its own arithmetic back to itself;
3. refuses a group whose registration is not `pixel`, whose transform is rotated, whose scale
   disagrees with its coordinates, whose shape disagrees with its arrays, or whose origin is
   *neither* the centre nor the corner — an unknown value is somebody else's change;
4. inspects every group before writing any, so a refusal on the last leaves the store untouched;
5. diffs the whole attribute dict before against after and aborts if anything outside
   `spatial:transform` and `zarr_conventions` moved — the provenance, run records and depth rule
   are in that dict, and `zarr` replaces attrs wholesale;
6. re-reads and re-verifies after committing, including that all 1,070 tags and the single branch
   survived;
7. is idempotent, and commits nothing when there is nothing to do.

Its test seeds a real Icechunk store, regresses it to the published shape, and asserts among other
things that the repaired attributes equal **what the fixed seeder writes from scratch, to the
float** — comparing the script against the source rather than against itself.

## 5. Three things deliberately not done

| considered | ruling |
|---|---|
| Flip `spatial:registration` to `"node"`, making the shipped transform correct by redefinition | **No.** The data is areal imagery, `"pixel"` is the honest description, and under `"node"` the bbox would have to become centre-based too — trading one contradiction for another and moving the published extent by half a pixel |
| Rewrite the 1,070 tags so tagged reads also see the fix | **No.** Tags are zone-year completion markers, not a consumer entry point; the recipe reads `main`. Rewriting a thousand refs to correct metadata nobody reads through them is risk without a reader |
| Chase the same attributes in `s3://tessera-embeddings/v1.1/cambridge.zarr` | **Out of scope.** Not written by this pipeline — different group naming, no `proj:`/`spatial:` at its root — and checked: its transform origin and bbox minimum already agree on the grid line, so it is corner-based and correct |

## 6. The one consequence to know about

`storage/conventions.py` sits inside the inference code-identity closure (46 files), so editing it
moves `inference_code_identity`. The campaign is complete and no staging prefix is waiting to be
resumed into, so nothing is invalidated — but a future repair fill will not reuse tiles staged
under the old identity. Recorded here because it is the kind of thing that should be a decision
rather than a discovery.
