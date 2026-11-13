# docs/ingest/

**How satellite ingestion works, one subject per document.** Start at the package README,
[`src/tessera_embeddings/ingest/README.md`](../../src/tessera_embeddings/ingest/README.md): it
walks the whole pipeline and summarises each subject below in a paragraph before linking here.

In the order the pipeline meets them:

| document | for anyone who… |
|---|---|
| [`zone-rois.md`](zone-rois.md) | runs the global campaign, or makes and checks the zone masks it ingests through |
| [`radar.md`](radar.md) | changes how OPERA radar is found and read, or asks which burst supplied a radar pixel |
| [`solar-days.md`](solar-days.md) | touches a date anywhere in the ingest, or asks why a day near the antimeridian came out short |
| [`earth-search-response-cap.md`](earth-search-response-cap.md) | changes the Sentinel-2 query, or sees a 502 from Earth Search |
| [`duplicate-copies.md`](duplicate-copies.md) | asks which copy of a reprocessed Sentinel-2 image was read, or why a date survived a read failure |
| [`baseline-correction.md`](baseline-correction.md) | asks whether and how the Sentinel-2 reflectance offset was removed from a pixel |
| [`provenance.md`](provenance.md) | asks which granules went into a mosaic slot, or which one supplied a pixel, or turns the record on |
| [`authentication.md`](authentication.md) | sets up Earthdata Login for radar, or diagnoses a credential error during an ingest |
| [`error-handling.md`](error-handling.md) | asks why an ingest failed, or what it does about a refused query or a corrupt file |
| [`performance.md`](performance.md) | asks what is limiting an ingest, and which levers to reach for |

Why these designs were chosen, and what was measured, is in
[`context_docs/ingest/`](../../context_docs/ingest/) and the decision records it links.
