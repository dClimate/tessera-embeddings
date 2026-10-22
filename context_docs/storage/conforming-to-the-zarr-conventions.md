# Conforming to the Zarr conventions: what the stores declare, how it is checked, and why

Every output store describes itself through the metadata conventions GeoZarr currently consists of,
plus the geoembeddings convention for the embeddings themselves. This records which versions we
follow, how conformance was verified, and the handful of places where following a convention meant
writing less rather than more. The code is `storage/conventions.py`; this was last audited in
PR #218 (2026-10-22).

## 1. What a store declares

| node | conventions registered | what it carries |
|---|---|---|
| root (global store) | `geoemb` | encoder, source data, quantization, `gsd`; once for all 120 zones |
| zone group, or root of a single-ROI store | `proj`, `spatial` (+ `geoemb` on a single-ROI root) | CRS as `proj:code`, `proj:wkt2` and `proj:projjson`; the grid as `spatial:dimensions`, `transform`, `shape`, `bbox`, `registration` |
| every array spanning both spatial dimensions | `spatial` | `spatial:dimensions` repeated from the group |
| `northing`, `easting` | none (CF attributes) | `standard_name`, `units: m`, `axis`, only for a CRS in metres |

| convention | version followed | latest upstream |
|---|---|---|
| [proj](https://github.com/zarr-conventions/proj) | v0.1 | v0.1 (`main` is identical) |
| [spatial](https://github.com/zarr-conventions/spatial) | v0.1 | v0.1 (`main` is identical) |
| [geoemb](https://github.com/geo-embeddings/embeddings-zarr-convention) | v1, as its schema names it | `main`; no tag has been cut |
| [zarr-conventions-spec](https://github.com/zarr-conventions/zarr-conventions-spec) | v1 | v1 |

[GeoZarr's own list](https://github.com/zarr-developers/geozarr-spec) is proj, spatial and
multiscales, with "CF in Zarr" still under consideration. We have no pyramids, so multiscales does
not apply yet.

## 2. How conformance is checked

Seed a two-zone global store and assemble a single-ROI store locally. Then validate every node's
`zarr.json` against the schema of each convention it registers, matched by `uuid`, and add the
cross-node checks a schema cannot express: `spatial:shape` against the coordinate arrays, and every
spatial array carrying its own `spatial:dimensions`. `proj:projjson` is validated separately against
[PROJJSON v0.7](https://proj.org/schemas/v0.7/projjson.schema.json), which the proj schema
references remotely.

**Use a JSON Schema 2020-12 validator, not the draft-07 the schemas declare.** Draft-07 ignores
every keyword beside a `$ref` ([proj#28](https://github.com/zarr-conventions/proj/issues/28)), and
that is exactly where the proj and spatial schemas check the `zarr_conventions` entry. A control
array registered as `"spatial:"`, and another with no registration at all, both pass draft-07 with
zero errors; 2020-12 rejects each. A corrupted `proj:projjson` is likewise rejected.

| store | built by main | built by PR #218 |
|---|---|---|
| two-zone global store | 21 problems | 0 |
| single-ROI store | 5 problems | 0 |

These counts cover the schemas and the cross-node checks. The decisions in section 3 that concern
meaning (which layout to claim, which source URLs to list) were checked by reading each
specification's prose.

## 3. Decisions

**The registration names stay `"proj"` and `"spatial"`, without a colon.** Upstream dropped the
colon on 2026-05-14 (spatial#27, proj#22), and both v0.1 schemas pin the bare name as a `const`.
rioxarray (0.22 and 0.23) still looks for `"spatial:"` and `"proj:"`, but the conventions spec says
a `name` "MUST NOT be used by tools to identify the Convention", so we follow the schema rather than
the reader. The published store matched rioxarray until ADR 024's repair corrected the names.

**The geoemb registration uses the `refs/tags/v1` URLs.** Its schema pins them as `const`, and v1 is
the convention's version. Upstream has not cut the tag, so the URLs 404 until it does. An earlier
commit-SHA URL resolved but failed the schema it pointed at.

**Every spatial array repeats `spatial:dimensions`.** The spatial spec requires it on arrays when
composed with `proj`, and a reader that opens one variable, such as an xarray `DataArray`, never
sees its group's attributes. The group keeps its own copy, which the geoembeddings spec asks of
per-zone stores.

**The coordinate arrays carry CF attributes, but CF is not registered.** `axis`, `standard_name` and
`units` are what CF-reading tools use to find the spatial axes; they are written only for a CRS in
metres, the same rule as `geoemb:gsd`. The [CF convention for Zarr](https://github.com/zarr-conventions/CF)
is a proposal with no release, and its registration example adds a `version` field the conventions
spec forbids, so registering it would not validate.

**`geoemb:spatial_layout` is not written.** The schema allows two values: `"utm_zones"`, defined as
60 groups named `utm01` to `utm60`, and `"global"`, one array in a global CRS. Our global store is
120 groups named `01N` to `60S`, each in its hemisphere's own EPSG code
([ADR 011](../decisions/011-campaign-zone-ingestion.md)), and a single-ROI store is neither. A tool
following the spec would look for `utm33` and not find it, so the optional field is omitted. The
convention's own TESSERA example instead extends the northern CRS south with negative northings. A
hemisphere amendment upstream would let us declare a layout again.

**`geoemb:source_data` lists resolvable dataset references:** the Earth Search `sentinel-2-l2a`
collection we query, which indexes both the `sentinel-cogs` and `sentinel-s2-l2a` buckets
([ADR 021](../decisions/021-correct-the-boa-offset-per-image.md)), and the OPERA RTC-S1 DOI. They
replace `s3://sentinel-cogs`, one of the two buckets, and an ASF datapool directory URL that 404s.

**`geoemb:model` for v1.1 still 404s** (`https://geotessera.org/model/1.1`); v2's Hugging Face URL
resolves. The spec asks only for a URI, so this is not a conformance failure. The value is part of
the store's identity (`check_root_identity`), so changing it in code would make the published v1.1
store refuse further fills. The fix is a page at that address.

## 4. What readers see

Measured with rioxarray 0.22 and odc-geo 0.5.1 on a seeded zone group:

| | built by main | built by PR #218 |
|---|---|---|
| rioxarray: spatial axes of a zone group / of one variable | neither found | both found |
| rioxarray: CRS of a zone group / of one variable | found / not found | found / not found |
| odc-geo: CRS of one variable | not found | not found |

The CRS lives on the zone group, which is where the proj spec recommends it, and xarray does not
pass a group's attributes to a variable ([rioxarray#888](https://github.com/vylor/rioxarray/issues/888)).
Readers write it themselves; the recipe is in
[reading the published store](reading-the-published-store.md#2a-georeferencing-where-a-pixel-actually-is).
Repeating `proj:` on every array, or adding a CF `grid_mapping` variable, would remove that step.

## 5. Upstream changes to watch before seeding v2

- **A single-source rule for `spatial:`** ([spatial#32](https://github.com/zarr-conventions/spatial/issues/32),
  [#23](https://github.com/zarr-conventions/spatial/issues/23)). Open, not released. It would put
  `dimensions`, `transform` and `shape` on arrays only, forbid `bbox` beside an affine transform, and
  leave groups only `bbox`. Our group-level transform and shape would then move to the arrays.
- **Versioning by `uuid`** ([zarr-conventions-spec#38](https://github.com/zarr-conventions/zarr-conventions-spec/issues/38)).
  Open. It may change how a registration names its version.
- **The geoemb `v1` tag**, which would make the registration URLs resolve.

## 6. The published v1.1 store

Its groups were seeded before this audit, and repairing them is deferred. It differs from what the
code now writes in five ways: the geoemb registration's commit-SHA URLs, no CF attributes on
`northing`/`easting`, no `spatial:dimensions` on the arrays, `geoemb:spatial_layout: "utm_zones"`,
and the earlier `geoemb:source_data`. All five are metadata. A repair would be one commit, made the
way [ADR 024](../decisions/024-spatial-transform-origin-is-the-pixel-corner.md) made its own.
