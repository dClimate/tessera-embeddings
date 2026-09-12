# docs/

**How to use this library.** Everything here answers "how do I run it, configure it, or plug it
into my own stack?"

If your question starts with *why* — why is the code shaped like this, why was that number chosen,
what was tried and abandoned — you want [`context_docs/`](../context_docs/) instead.

## Which one do I want?

| you want to… | go to |
|---|---|
| run the pipeline for the first time | [`quickstart.md`](quickstart.md) |
| know what the configuration objects hold, and what each field does | [`configuration.md`](configuration.md) |
| get a working Python environment | [`environment-setup.md`](environment-setup.md) |
| stand up the reference orchestrator | [`prefect-setup.md`](prefect-setup.md) |
| run it on AWS | [`providers/aws.md`](providers/aws.md) |
| run it on something other than AWS | [`providers/adding-your-own.md`](providers/adding-your-own.md) |
| use a different orchestrator than Prefect | [`orchestrator-swap.md`](orchestrator-swap.md) |
| know which names are stable to import | [`public-api.md`](public-api.md) |
| choose between one area and the whole globe, or read the published global store | [`single-vs-global.md`](single-vs-global.md) |
| understand *why* a design is the way it is | [`../context_docs/`](../context_docs/) |
| understand what the code *is* | the [top-level README](../README.md) |

## The difference, in one line each

**`docs/` is reference.** It describes the current state of the software and is expected to be
correct. If something here is wrong or stale, that is a bug — please say so.

**`context_docs/` is the record.** It holds decision records ("why is X the way it is?") and stage
records ("what did we measure, what did it cost, what did we try that failed?"). It is deliberately
historical: it keeps superseded figures next to the corrections that replaced them, because a
number you can still see, with its correction attached, cannot be quoted by accident. It is not a
description of how to use anything, and some of it describes code that no longer exists.

**The top-level [README](../README.md) is the overview** — what the library does, how it is
layered, and how the global embeddings store is shaped. Start there if you have not used the
project before.

## A note on numbers

Measurements do not belong in this directory. If you find a throughput figure, a cost, or a
benchmark here, it is either a short summary with a link, or it has drifted from
`context_docs/` and should be replaced by one. The reason is that a figure quoted in two places
gets corrected in one of them — which has happened often enough in this repository to have its own
index, [`context_docs/corrections-register.md`](../context_docs/corrections-register.md).
