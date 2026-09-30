"""What the global campaign actually cost, from measured usage priced at on-demand list rates.

Produces the cost half of ``context_docs/campaign/campaign-cost-model.md`` section 12: total
instance-hours, container-hours, storage and request volumes over the campaign window, each
multiplied by the published list price for its exact billing usage type.

    python scripts/scoping/campaign_cost_actuals.py --start 2026-07-01 --end 2026-09-11

``--end`` IS EXCLUSIVE, which is Cost Explorer's convention and the single easiest thing to get
wrong here: the command above closes the window on 2026-09-10, which is what §12 publishes. An
``--end`` of 2026-09-12 would pull in a day of usage that is not campaign work and a day Cost
Explorer had not finished reporting.

WHAT IT PRODUCES IS NOT A BILL, and the distinction is the reason this script exists rather
than a console screenshot. Cost Explorer's *cost* metrics read as zero in this member account
-- ``UnblendedCost`` returns fractions of a cent for days that certainly cost five figures --
so no dollar figure can be reconciled from here. ``UsageQuantity`` is intact, and reconciles
against itself: the daily instance-hours sum exactly to the monthly totals. So usage is the
measurement and the price is published list. Savings plans, reserved capacity, credits and
enterprise discounts are invisible from here and can only reduce the result.

FOUR ASSUMPTIONS THAT WOULD SILENTLY CORRUPT THE ANSWER, each guarded below:

* ``UsageQuantity`` grouped without a usage type mixes hours with gigabyte-months and request
  counts, and the API then reports the unit as "N/A". Everything is keyed on the usage-type
  string, and the unit is asserted before the quantity is used.
* S3 storage steps down in price at 50 TB and 500 TB, so it is priced per calendar month
  against its own tiers. A campaign this size priced at the first tier is overstated.
* The most recent day or two is incomplete, because Cost Explorer trails real time -- which
  understated this campaign by $11,000 once (§12). The last day in the data is reported so it is
  not read as a fall in activity, and Cost Explorer's own ``Estimated`` flag is surfaced per
  period. Note what that flag does and does not mean: it marks the whole of a month AWS has not
  closed for billing, not a day whose usage is still arriving. It is therefore a statement about
  finality and not about completeness, and both matter.
* Anything with no price entry is EXCLUDED and printed IN FULL, with no quantity threshold.
  A threshold cannot be set here, because quantities are not comparable across billing units:
  900 hours of a large instance is five figures and 900 S3 requests is nothing. Unpriced EC2
  instance-hours are called out separately, since those are the ones that are large money at
  small quantity.

Re-run this at the end of a campaign, and widen ``EC2_TYPES`` if the fleet gains an instance type;
an unpriced type shows up in the excluded list rather than vanishing.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Any

import boto3

#: Cost Explorer and the Pricing API are both global endpoints served from these regions.
CE_REGION = "us-east-1"
PRICING_REGION = "us-east-1"
PRICING_LOCATION = "US West (Oregon)"

#: EC2 instance types to price, keyed by the Cost Explorer usage type they bill under.
EC2_TYPES = {
    "USW2-BoxUsage:g5.2xlarge": "g5.2xlarge",
    "USW2-BoxUsage:g6e.xlarge": "g6e.xlarge",
    "USW2-BoxUsage:m5.2xlarge": "m5.2xlarge",
    "USW2-BoxUsage:c8gn.48xlarge": "c8gn.48xlarge",
}
#: Everything else, with the price fetched by usage type rather than by instance type.
FLAT_TYPES = {
    "USW2-Fargate-vCPU-Hours:perCPU": ("AmazonECS", "hours"),
    "USW2-Fargate-GB-Hours": ("AmazonECS", "hours"),
    "USW2-Requests-Tier1": ("AmazonS3", "Requests"),
    "USW2-Requests-Tier2": ("AmazonS3", "Requests"),
    "USW2-Inventory-ObjectsListed": ("AmazonS3", "Objects"),
    "USW2-EBS:VolumeUsage.gp3": ("AmazonEC2", "GB-Mo"),
}
#: Priced per calendar month against its own tier boundaries, which are FETCHED and not pinned --
#: see :func:`storage_tiers`. A campaign this size sits in the third tier, so hard-coding the first
#: rate overstates it, and hard-coding all three would go stale silently against every other line
#: on this page being live.
STORAGE_TYPE = "USW2-TimedStorage-ByteHrs"

#: How the lines are reported: (label, usage types, is_campaign_work). Order is presentation order.
#:
#: The third field exists because the account carries spend that is not the campaign, and a script
#: whose only total silently includes it invites that total to be quoted as the campaign's cost.
#: Both subtotals are printed; only the campaign one is comparable with the cost model.
GROUPS = (
    ("graphics cards", ("USW2-BoxUsage:g5.2xlarge", "USW2-BoxUsage:g6e.xlarge"), True),
    ("Fargate containers", ("USW2-Fargate-vCPU-Hours:perCPU", "USW2-Fargate-GB-Hours"), True),
    ("S3 storage", (STORAGE_TYPE,), True),
    ("S3 requests", ("USW2-Requests-Tier1", "USW2-Requests-Tier2", "USW2-Inventory-ObjectsListed"), True),
    ("EBS volumes", ("USW2-EBS:VolumeUsage.gp3",), True),
    ("Ray head nodes", ("USW2-BoxUsage:m5.2xlarge",), True),
    # The isolated VPC's three NAT instances, 2026-08-20 to 2026-09-11. Real spend, not campaign
    # production, and kept as its own line so it can neither be hidden nor double-counted.
    ("isolated-VPC NAT instances", ("USW2-BoxUsage:c8gn.48xlarge",), False),
)


def ec2_price(pricing: Any, instance_type: str) -> float:  # noqa: ANN401 — botocore client, untyped
    """On-demand Linux hourly price. The filter set matters: the same instance type carries a
    dozen price dimensions, and omitting any one of these returns several and picks arbitrarily.
    """
    resp = pricing.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": f, "Value": v}
            for f, v in (
                ("instanceType", instance_type),
                ("location", PRICING_LOCATION),
                ("operatingSystem", "Linux"),
                ("tenancy", "Shared"),
                ("preInstalledSw", "NA"),
                ("capacitystatus", "Used"),
            )
        ],
        MaxResults=10,
    )
    prices = {
        float(dim["pricePerUnit"]["USD"])
        for blob in resp["PriceList"]
        for term in json.loads(blob)["terms"].get("OnDemand", {}).values()
        for dim in term["priceDimensions"].values()
        if dim["unit"] == "Hrs"
    }
    if not prices:
        raise SystemExit(f"no on-demand hourly price for {instance_type} — refusing to guess one")
    return min(prices)


def flat_price(pricing: Any, service: str, usage_type: str) -> float:  # noqa: ANN401 — botocore client, untyped
    """Price for one usage type, matched on the usage type the bill uses.

    Matched that way rather than on a description because these services carry many dimensions
    whose descriptions overlap, and the usage type is the only unambiguous key.
    """
    prices: set[float] = set()
    token = None
    while True:
        kw: dict[str, Any] = {
            "ServiceCode": service,
            "Filters": [{"Type": "TERM_MATCH", "Field": "usagetype", "Value": usage_type}],
            "MaxResults": 100,
        }
        if token:
            kw["NextPageToken"] = token
        resp = pricing.get_products(**kw)
        for blob in resp["PriceList"]:
            for term in json.loads(blob)["terms"].get("OnDemand", {}).values():
                for dim in term["priceDimensions"].values():
                    prices.add(float(dim["pricePerUnit"]["USD"]))
        token = resp.get("NextPageToken")
        if not token:
            break
    if not prices:
        raise SystemExit(f"no price for usage type {usage_type} — refusing to guess one")
    return min(prices)


def storage_tiers(pricing: Any) -> tuple[tuple[float, float], ...]:  # noqa: ANN401 — botocore client, untyped
    """S3 Standard storage price tiers as ``((upper_gb_month, rate), ...)``, ascending.

    Fetched rather than pinned. The Pricing API returns one price dimension per tier carrying its
    own ``beginRange``/``endRange``, so the boundaries are published data like every other rate on
    this page; pinning them would leave one line of the table frozen while the rest moved.
    """
    resp = pricing.get_products(
        ServiceCode="AmazonS3",
        Filters=[{"Type": "TERM_MATCH", "Field": "usagetype", "Value": STORAGE_TYPE}],
        MaxResults=100,
    )
    tiers: list[tuple[float, float]] = []
    for blob in resp["PriceList"]:
        for term in json.loads(blob)["terms"].get("OnDemand", {}).values():
            for dim in term["priceDimensions"].values():
                if dim["unit"] != "GB-Mo":
                    continue
                upper = dim.get("endRange") or "Inf"
                tiers.append(
                    (float("inf") if upper in ("Inf", "") else float(upper), float(dim["pricePerUnit"]["USD"]))
                )
    if not tiers:
        raise SystemExit(f"no storage tiers for {STORAGE_TYPE} — refusing to guess them")
    return tuple(sorted(tiers))


def fetch_usage(ce: Any, start: str, end: str) -> tuple[dict, dict, list[str]]:  # noqa: ANN401 — botocore client, untyped
    """(usage[day][usage_type], unit[usage_type], periods AWS still calls estimated).

    Paginated deliberately: a single page is not the whole window, and a truncated fetch is
    indistinguishable from a quiet campaign.

    The ``Estimated`` flag is carried out rather than dropped. It does not mean what the trailing-day
    problem needs it to mean -- AWS sets it for every day of an unclosed month, not for a day whose
    usage is still landing -- so it cannot be the guard against closing a window too early. It is
    still the only statement the API makes about whether a figure is final, and a caller quoting a
    total deserves to know which part of it AWS has not closed.
    """
    usage: dict[str, dict[str, float]] = collections.defaultdict(dict)
    units: dict[str, str] = {}
    estimated: list[str] = []
    token, pages = None, 0
    while True:
        kw: dict[str, Any] = {
            "TimePeriod": {"Start": start, "End": end},
            "Granularity": "DAILY",
            "Metrics": ["UsageQuantity"],
            "GroupBy": [{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
        }
        if token:
            kw["NextPageToken"] = token
        resp = ce.get_cost_and_usage(**kw)
        pages += 1
        for period in resp["ResultsByTime"]:
            day = period["TimePeriod"]["Start"]
            if period.get("Estimated") and day not in estimated:
                estimated.append(day)
            for group in period["Groups"]:
                ut = group["Keys"][0]
                metric = group["Metrics"]["UsageQuantity"]
                usage[day][ut] = usage[day].get(ut, 0.0) + float(metric["Amount"])
                units[ut] = metric["Unit"]
        token = resp.get("NextPageToken")
        if not token:
            break
    print(f"fetched {pages} page(s): {len(usage)} days, {len(units)} usage types")
    return usage, units, sorted(estimated)


def storage_cost(usage: dict, tiers: tuple[tuple[float, float], ...]) -> tuple[float, dict[str, tuple[float, float]]]:
    """S3 storage priced per calendar month against its tier boundaries."""
    by_month: collections.Counter = collections.Counter()
    for day, day_usage in usage.items():
        if STORAGE_TYPE in day_usage:
            by_month[day[:7]] += day_usage[STORAGE_TYPE]
    total, detail = 0.0, {}
    for month, gb_months in sorted(by_month.items()):
        remaining, cost, floor = gb_months, 0.0, 0.0
        for bound, rate in tiers:
            take = min(remaining, bound - floor)
            if take <= 0:
                break
            cost += take * rate
            remaining -= take
            floor = bound
        detail[month] = (gb_months, cost)
        total += cost
    return total, detail


def main(argv: list[str] | None = None) -> int:
    """Print the campaign's measured usage and its list-price cost. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="inclusive, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="EXCLUSIVE, YYYY-MM-DD (the Cost Explorer convention)")
    ap.add_argument(
        "--require-final",
        action="store_true",
        help="refuse if AWS still calls any period in the window estimated (i.e. an unclosed month)",
    )
    args = ap.parse_args(argv)

    session = boto3.Session()
    ce = session.client("ce", region_name=CE_REGION)
    pricing = session.client("pricing", region_name=PRICING_REGION)

    usage, units, estimated = fetch_usage(ce, args.start, args.end)
    if not usage:
        print("no usage in that window", file=sys.stderr)
        return 1
    if estimated:
        print(
            f"\nAWS still calls {len(estimated)} period(s) ESTIMATED: {estimated[0]} .. {estimated[-1]}."
            "\n  That flag marks an unclosed BILLING month, not a day whose usage is still arriving,"
            "\n  so it is a statement about finality rather than completeness. Quantities in an"
            "\n  unclosed month can still move.",
            file=sys.stderr,
        )
        if args.require_final:
            print("REFUSING: --require-final was passed. Move --end back to a closed month.", file=sys.stderr)
            return 1

    price = {ut: ec2_price(pricing, it) for ut, it in EC2_TYPES.items()}
    price.update({ut: flat_price(pricing, svc, ut) for ut, (svc, _) in FLAT_TYPES.items()})
    tiers = storage_tiers(pricing)
    print("\nS3 storage tiers (fetched): " + ", ".join(f"<={b:,.0f} GB-Mo ${r}" for b, r in tiers))
    print("\nunit prices (on-demand list, fetched now):")
    for ut, p in sorted(price.items()):
        print(f"  {ut:<40} ${p:.6f} per {units.get(ut, '?')}")

    quantity: collections.Counter = collections.Counter()
    for day_usage in usage.values():
        for ut, q in day_usage.items():
            quantity[ut] += q

    # Assert the unit before multiplying: an hours price against a gigabyte-month quantity is a
    # wrong answer that looks plausible.
    for ut in price:
        if ut in quantity and units.get(ut) not in ("Hrs", "hours", "Requests", "Objects", "GB-Month", "GB-Mo"):
            print(f"  REFUSING {ut}: unexpected unit {units.get(ut)!r}", file=sys.stderr)
            return 1

    stor_total, stor_detail = storage_cost(usage, tiers)

    days = sorted(usage)
    gpu_uts = GROUPS[0][1]
    gpu_days = [d for d in days if sum(usage[d].get(u, 0.0) for u in gpu_uts) > 1]
    if gpu_days:
        print(f"\ngraphics-card days: {gpu_days[0]} .. {gpu_days[-1]} ({len(gpu_days)} days)")

    print(f"\n{'line':<26} {'quantity':>34} {'$ at list':>13}")
    campaign, other = 0.0, 0.0
    for label, uts, is_campaign in GROUPS:
        if label == "S3 storage":
            cost, qty = stor_total, f"{quantity[STORAGE_TYPE]:,.0f} GB-Mo"
        else:
            cost = sum(quantity[u] * price[u] for u in uts)
            qty = " + ".join(f"{quantity[u]:,.0f}" for u in uts)
        print(f"{label:<26} {qty:>34} {cost:>13,.0f}")
        # Summed as PRINTED, to the dollar, so the table adds up for whoever checks it by hand and
        # so a figure quoted from here matches a figure quoted from the printed lines.
        if is_campaign:
            campaign += round(cost)
        else:
            other += round(cost)
    print(f"{'CAMPAIGN TOTAL':<26} {'':>34} {campaign:>13,.0f}")
    if other:
        print(f"{'GRAND TOTAL — incl. non-campaign':<26} {'':>34} {campaign + other:>13,.0f}")

    print("\n  S3 storage by month (GB-months -> $):")
    for month, (gb, cost) in stor_detail.items():
        print(f"    {month}  {gb:>14,.0f}  ${cost:>12,.0f}")

    # Printed in full and unfiltered. A quantity threshold here would be a comparison across
    # billing units -- hours against requests against gigabyte-months -- and the expensive omission
    # is precisely the small-quantity, high-price one a threshold hides.
    priced = set(price) | {STORAGE_TYPE}
    unpriced = [(ut, q) for ut, q in sorted(quantity.items(), key=lambda kv: -kv[1]) if ut not in priced]
    print(f"\nEXCLUDED — every usage type with no price entry here ({len(unpriced)}):")
    for ut, q in unpriced:
        print(f"  {ut:<44} {q:>18,.2f} {units[ut]}")
    boxes = [(ut, q) for ut, q in unpriced if ":" in ut and "BoxUsage" in ut]
    if boxes:
        print("\n  *** UNPRICED EC2 INSTANCE-HOURS — these are the ones that are large money at")
        print("      small quantity. Add each to EC2_TYPES, or satisfy yourself it is not ours:")
        for ut, q in boxes:
            print(f"        {ut:<42} {q:>14,.0f} h")

    print(f"\nlast day in the data: {days[-1]} — INCOMPLETE, Cost Explorer trails real time")
    print("These are LIST PRICES on measured usage, not a bill. See this module's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
