"""Inspect archived CAISO chart CSV schemas without inferring service reliability.

Usage:
  python external_schema_check.py --data-dir data_raw_external/caiso \
      --output-dir reproduced_results/external_schema

Only Python's standard library is required. The Pacific calendar is an explicit
interpretive convention. No missing interval is imputed and no clock-only record
is silently assigned to one of the two offsets in an ambiguous local hour.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]

COLUMNS = ["Time", "Day ahead forecast", "Hour ahead forecast", "Current demand", "Demand response"]


def valid_instants(local: datetime, zone: ZoneInfo) -> set[datetime]:
    """Return distinct UTC instants whose round trip preserves a naive clock."""
    candidates = set()
    for fold in (0, 1):
        instant = local.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == local:
            candidates.add(instant)
    return candidates


def inspect_csv(path: Path, zone_name: str) -> dict:
    day = date.fromisoformat(path.stem.removeprefix("demand_"))
    zone = ZoneInfo(zone_name)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected schema in {path.name}: {reader.fieldnames}")
        rows = list(reader)
    clocks = [row["Time"] for row in rows]
    counts = Counter(clocks)
    repeated = {clock: count for clock, count in counts.items() if count > 1}
    nonexistent = []
    ambiguous = []
    for clock in sorted(counts):
        hour, minute = map(int, clock.split(":"))
        local = datetime.combine(day, time(hour, minute))
        cardinality = len(valid_instants(local, zone))
        if cardinality == 0:
            nonexistent.append(clock)
        elif cardinality > 1:
            ambiguous.append(clock)
    start = datetime.combine(day, time(), tzinfo=zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time(), tzinfo=zone).astimezone(timezone.utc)
    expected_slots = int((end - start).total_seconds() // 300)
    blank_counts = {column: sum(not row[column].strip() for row in rows) for column in COLUMNS[1:]}
    blank_clocks = {column: [row["Time"] for row in rows if not row[column].strip()] for column in COLUMNS[1:4]}
    all_three_calendar_aligned = bool(nonexistent) and all(sorted(items) == nonexistent for items in blank_clocks.values())
    disposition = "Explicit boundary and timestamp convention required before UTC window audit"
    if ambiguous:
        disposition = "Repeated-hour offset unresolved; defer UTC window audit"
    elif all_three_calendar_aligned:
        disposition = "Blank load clocks match nonexistent local hour; do not label as service omissions"
    return {
        "date": day.isoformat(),
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "published_data_rows": len(rows),
        "distinct_clock_labels": len(counts),
        "repeated_clock_labels": repeated,
        "first_clock": clocks[0],
        "last_clock": clocks[-1],
        "expected_five_minute_slots_in_local_day": expected_slots,
        "calendar_timezone": zone_name,
        "calendar_nonexistent_clock_labels": nonexistent,
        "calendar_ambiguous_clock_labels": ambiguous,
        "blank_counts": blank_counts,
        "blank_load_clocks_match_nonexistent_hour": all_three_calendar_aligned,
        "utc_offset_or_fold_column_present": False,
        "revision_or_publication_time_column_present": False,
        "disposition": disposition,
        "window_availability_estimated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_raw_external" / "caiso")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_results" / "external_schema")
    parser.add_argument("--timezone", default="America/Los_Angeles")
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    results = []
    for source in manifest["files"]:
        path = args.data_dir / source["filename"]
        result = inspect_csv(path, args.timezone)
        if result["sha256"] != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        results.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Three deliberately selected source-format examples; no regional reliability or model-performance estimate",
        "timezone_status": "Pacific-time calendar interpretation based on the publisher's PT display and published chart code; CSV rows contain no offset",
        "midnight_status": "The final repeated 00:00 row is retained as a boundary ambiguity, not silently removed or assigned to a date",
        "demand_response_status": "Blank Demand response fields are reported separately; they are not treated as missing demand observations or as zeros",
        "results": results,
    }
    (args.output_dir / "caiso_schema_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fieldnames = ["date", "published_data_rows", "distinct_clock_labels", "expected_five_minute_slots_in_local_day", "blank_current_demand", "nonexistent_clock_labels", "ambiguous_clock_labels", "disposition"]
    with (args.output_dir / "caiso_schema_check.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "date": result["date"],
                "published_data_rows": result["published_data_rows"],
                "distinct_clock_labels": result["distinct_clock_labels"],
                "expected_five_minute_slots_in_local_day": result["expected_five_minute_slots_in_local_day"],
                "blank_current_demand": result["blank_counts"]["Current demand"],
                "nonexistent_clock_labels": len(result["calendar_nonexistent_clock_labels"]),
                "ambiguous_clock_labels": len(result["calendar_ambiguous_clock_labels"]),
                "disposition": result["disposition"],
            })
    print(json.dumps({"files_checked": len(results), "output_dir": str(args.output_dir), "availability_estimated": False}))


if __name__ == "__main__":
    main()
