# Run identity, staged work, and how a resume decides what to reuse

**The question this answers.** A fill stages every inference tile to S3 before assembling it into
the store, and a re-run has to decide whether the tiles already there describe the same work it is
about to do. Getting that wrong in either direction is expensive: reuse tiles produced by different
code and one write-once zone-year holds two versions of the data; abandon good tiles and a re-run
pays for inference it already has.

The rule the campaign runs on is a **fingerprint**. A run's staging prefix is derived from what the
work IS — the inputs, the parameters and the code that turns one into the other — so two runs share
a prefix only when they would produce the same tiles.

**Nothing here relaxes a gate on the published store.** All of it decides which staging prefix a run
reads and writes. A cell's completion mark, its write-once zone-year tag and its manifest checks are
untouched by every lever below.

Operational summary in `campaign-plan.md` §3 (the settings) and §8 (failure handling); this file
holds the mechanism and the failure modes it is shaped by.

---

## 1. What the fingerprint covers

| component | why it is in the hash |
|---|---|
| the input mosaic's identity | different pixels in, different pixels out |
| the coverage threshold and the S1 orbit | they change which tiles exist and what each one holds |
| `allow_s2_only` | it changes which PIXELS are embedded, not just which tiles |
| the inference source hash (`inference_code_identity`) | the code that turns input into output |

**The code component is the inference source only.** An orchestration or ingest fix therefore reuses
staged tiles rather than abandoning them. Two overrides exist for the judgement a hash cannot make
(§3), and the AMI is resolved once per campaign and pinned into every fill, which is the only thing
stopping one run straddling two images.

## 2. Three properties that are easy to reintroduce a hole in

Each is a path around a check meant to be unavoidable, and each was one.

**The import closure resolves relative imports.** A relative import's module name is the tail alone
(`providers` for `from .providers import …`), so a walker that follows only names starting with the
package stops at every `__init__`. That left `config/providers.py` — the STAC collections, band
lists, resolutions and baseline settings — *outside* the ingest fingerprint, so changing which
imagery a mosaic is built from left its identity unmoved and a resume was free to append across the
change. The ingest closure is 30 files: enumerate it, and never trust a docstring's account of it.

**The completion marker validates identity in its own pass.** The ingest code identity is checked on
the *append* path, so that a code change does not declare finished mosaics stale. But a resume
adopting a store whose every date already landed appends nothing, and with no append there is no
check — the marker would land over a mosaic built under a different mask, threshold or code, after
which every later run skips the cell. Validating before marking is what stops a disagreement on the
last store leaving the earlier ones marked.

**`ABSENT_MEANS_OFF` names the fields where having no opinion is itself an opinion.**
`allow_s2_only` records "off" by being absent, for compatibility with stores predating the field. So
manifest validation that skips absent keys let a run with the flag on pass against a legacy store
and mix both pixel policies in one store — the exact append the field exists to refuse. A field
*describing* a store and a field stating the *policy* it was built under want opposite answers, and
one rule cannot express both.

## 3. Reaching existing staging: what works, and what only looks like it does

**`staging_code_identity` is the campaign-level lever, and `run_id` the per-cell one.** Pass
`run_global_campaign`'s `staging_code_identity` to state the fingerprint's code component outright
instead of deriving it, or `fill-zone-year`'s explicit `run_id` to resume one specific prefix.
Everything else changes the prefix as a side effect of changing the fingerprint.

**This paragraph used to say `run_id` was the only reliable lever, and that was true when it was
written.** It is still the only one for a single cell, but a campaign dispatches its own children
and derives their run ids, so there was no way to restart a campaign onto tiles it had already
staged. The distinction that matters is DERIVED versus STATED: both `force_staging_*` hatches
compute the identity from the code in front of them, and no computation over changed code can
reproduce the identity that changed code replaced. Only stating it reaches backwards.

The judgement `staging_code_identity` rests on is exactly `force_staging_reuse`'s — that the change
cannot alter what a staged tile CONTAINS. What makes a restart across an orchestration fix a fair
case is that the fingerprint covers the whole inference closure, so a change to how many actors are
requested, to an initialisation timeout, or to a preflight coverage gate abandons every staged tile
exactly as a change to the model would. On a deployment that ships a source tarball it is worse
still: the tarball's ETag is a term in the identity, so re-uploading it moves the fingerprint
whatever the change was, and the code comment calling that term "empty in production" assumes a
baked-AMI deploy that the global campaign is not running.

The three levers are mutually exclusive and refused together at preflight rather than ordered by
precedence, because each of them CLAIMS the identity and silently preferring one lands the run on a
prefix nobody chose.

**What it cannot be used for.** A change to the model, the checkpoint, or the pixel maths. The
per-tile validation in `StagedShardSource` catches the coarse form of a wrong judgement — a missing
variable, a wrong shard extent, a dtype mismatch, each failing loudly and naming the tile — but
same-shape-different-numbers passes it, and lands two code versions in one write-once zone-year.

**`force_staging_reuse` cannot reach staging created without it.** It substitutes a constant into the
hash, so it yields a *different* prefix from the one an unflagged run staged under; it preserves
reuse only between runs that both set it. Set it when a change to the inference source provably
cannot alter staged output — a log line, a comment, a type annotation — and the staging hours are
worth having. It is unsafe if that judgement is wrong, because `assemble_global` probes the variable
set from a single tile on the assumption that a prefix is homogeneous.

**`force_staging_restage` is the safe one.** Any new token starts a fresh prefix, for a change the
source hash cannot see — a deliberate dependency upgrade, where a new torch changes the numbers
without changing our source. Abandoning stale staged work is always safe, so unlike
`force_staging_reuse` this is usable in production.

**`fill-zone-year` mints a fresh `run_id` when the parameter is omitted.** The deterministic staging
fingerprint belongs to the campaign driver, not the per-cell flow, so a bare re-dispatch of the cell
flow starts staging over.

**Re-assembling an already-complete zone-year needs `assemble_global` called directly** with the
preserved run id. Both flow paths refuse a complete cell; the direct call is repeatable because it
never moves the tag.

## 4. The prefix is the record of what the staged pixels are

The `run_id` is minted from the **effective** S2-only mode — `InferenceConfig` forces
`allow_s2_only` when the orbit resolves to none — rather than from the requested flag, and
assembly-only mode reads the policy off the run_id prefix rather than trusting a parameter nobody had
to set. A parameter describes an intention; the prefix describes what is on disk.

## 5. A change inside the fingerprint closure has a landing window

Such a change moves the fingerprint, so a mosaic caught **mid-append** refuses its next append and
needs either `allow_ingest_code_mismatch` (off by default; relaxes only the code-identity term and
records both identities on the store) or its interrupted store deleted by hand. Finished mosaics are
unaffected. `config/providers.py` and the ingest query modules are inside that closure, so
**anything touching the query, the collections or the provider settings is a between-campaigns
change unless the override is used.** Land one otherwise only after confirming nothing is
mid-ingest: no ingest runner and no fleet up, only the Prefect worker service.
