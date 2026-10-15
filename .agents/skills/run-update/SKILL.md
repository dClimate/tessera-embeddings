---
name: run-update
description: Run a targeted update of a tessera-embeddings store — re-infer chosen tiles of windows or zone-years the store already holds, either filling only the pixels that have no embedding (holes) or rewriting whole tiles (shards). Use when asked to fill holes, take in new imagery, redo tiles after a fix, or update part of a single-area or global store.
---

# Running an update

The procedure is [`docs/updates.md`](../../../docs/updates.md). Read it first and follow it step by
step. This skill adds only what an agent must do on top of it.

## Before anything is written

- **Agree the name, the mode and the tiles with the person.** A name runs once, and a written plan
  cannot be changed.
- **Show the plan's dry run** (`plan_update.py` without `--write`) and get a yes before writing it.
- **Get a yes before starting a run.** It provisions GPUs and writes to the store.
- **Lowering a store's depth rule is a write of its own** (the guide's "Filling pixels a looser rule
  now admits"): show its dry run and get a yes before `--write`.
- **Check that nothing else is updating the same zone-years.** Two updates of one tile conflict.

## While it runs

On a failure, match the message against the guide's "If something goes wrong" table. Running the
same command again resumes; do not write a new plan to work around a failed run.

## Before calling it done

Do every check in the guide's "Check it" step: every zone-year `done`, the complete tag present, the
diff touching only the plan's tiles, and the registry `verify` passing. For a `holes` update, also
confirm on one written tile that every pixel whose `scales` was not NaN at the base tag is
byte-identical at the complete tag in all eight arrays. Report what you checked and what you found.

## Only with the person's say-so

- Delete the base tag, expire or garbage-collect the store, or reset a branch.
- Replace a mistaken plan with a new one under another name.
