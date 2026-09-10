"""Static checks on the ingest Logs-Insights query pack.

These exist because the plumbing tests around ``insights_query`` use a fake logs
client, which validates the poll/error contract but never the **query language** —
so four of the eight queries shipped syntactically invalid and always returned
``rows: null``. CloudWatch rejected them with
``MalformedQueryException: unexpected symbol found (`` and the tool, correctly,
just recorded the failure and moved on. The pack looked fine in CI and was half
dead against the real service.

Only a live CloudWatch call can fully validate Insights grammar, so these are the
cheap offline guards for the mistakes actually made. Anything subtler needs a run
against the service (see the README's per-run workflow).
"""

from __future__ import annotations

import re

import pytest

from tessera_embeddings.profiling.ingest.ingest_log_queries import QUERIES


@pytest.mark.parametrize("name", sorted(QUERIES))
def test_no_sort_on_a_raw_bin_expression(name: str) -> None:
    """``sort bin(5m)`` is invalid: a bin must be aliased, then the alias sorted.

    Insights parses ``sort`` over field names, not function calls, so
    ``stats ... by bin(5m) | sort bin(5m) asc`` is rejected outright. The correct
    form is ``stats ... by bin(5m) as period | sort period asc``.
    """
    _, query = QUERIES[name]
    assert not re.search(r"\bsort\s+bin\s*\(", query), (
        f"{name}: sorts a raw bin() expression — alias it (`by bin(5m) as period | sort period asc`)"
    )


@pytest.mark.parametrize("name", sorted(QUERIES))
def test_binned_queries_alias_and_sort_that_alias(name: str) -> None:
    """A time-binned query must alias its bin and sort by the alias.

    Without the sort the row order is unspecified, which matters here: these
    series are read as curves over a run (is the retry rate climbing?), so an
    unordered result is misleading rather than merely untidy.
    """
    _, query = QUERIES[name]
    if "by bin(" not in query:
        return
    m = re.search(r"by\s+bin\([^)]*\)\s+as\s+(\w+)", query)
    assert m, f"{name}: groups by bin() without aliasing it"
    assert re.search(rf"\bsort\s+{m.group(1)}\b", query), (
        f"{name}: aliases the bin as {m.group(1)!r} but never sorts by it"
    )


@pytest.mark.parametrize("name", sorted(QUERIES))
def test_sorted_fields_are_fields_the_query_defines(name: str) -> None:
    """Every ``sort`` target must be something the query actually produced.

    Catches the general shape of the bug above: sorting a name that no ``stats``
    alias, ``parse`` capture, or ``fields`` selection ever created.
    """
    _, query = QUERIES[name]
    sorted_fields = re.findall(r"\bsort\s+([A-Za-z_]\w*)", query)
    if not sorted_fields:
        return
    defined = set(re.findall(r"\bas\s+(\w+)", query))
    defined |= set(re.findall(r"\(\?<(\w+)>", query))  # parse captures
    for field in sorted_fields:
        assert field in defined, f"{name}: sorts by {field!r}, which the query never defines"


def test_every_query_has_a_description() -> None:
    """--list is the discovery surface; an undescribed query is unusable."""
    for name, (description, _) in QUERIES.items():
        assert description.strip(), f"{name}: empty description"
