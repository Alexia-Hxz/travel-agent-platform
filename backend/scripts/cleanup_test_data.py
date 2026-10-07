"""Remove fixture trips that test runs created, keeping the user's own trips.

Every test fixture names its trip with a unique 8 character hex suffix, for example
``行程测试 4f2a91cc``, ``file upload test 0d3b77ea`` or ``agent adjustment 91ff02ba``.
Trips the user created by hand are named like ``上海国庆游`` and never carry that
suffix, so the suffix is a reliable marker for machine created data.

Usage::

    uv run python scripts/cleanup_test_data.py            # report only, no changes
    uv run python scripts/cleanup_test_data.py --apply    # delete the fixture trips

``--apply`` also removes the places, itinerary items, uploaded files and
adjustments that belong to the deleted trips, and deletes the uploaded files from
disk. It never touches a trip without the fixture marker.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_DB = BACKEND_ROOT / "travel_agent.db"
CHILD_TABLES = ("places", "itinerary_items", "trip_files", "itinerary_adjustments")
FIXTURE_SUFFIX = re.compile(r"[0-9a-f]{8}$")
FIXTURE_DESTINATION = "AgentTestCity"


def is_fixture(title: object, destination: object) -> bool:
    """True when a trip was created by the test suite rather than by the user."""
    if str(destination or "").strip() == FIXTURE_DESTINATION:
        return True
    return bool(FIXTURE_SUFFIX.search(str(title or "").strip()))


def find_fixture_trips(connection: sqlite3.Connection) -> list[tuple[int, str, str]]:
    rows = connection.execute("SELECT id, title, destination FROM trips ORDER BY id").fetchall()
    return [
        (int(row[0]), str(row[1] or ""), str(row[2] or ""))
        for row in rows
        if is_fixture(row[1], row[2])
    ]


def keep_trips(connection: sqlite3.Connection) -> list[tuple[int, str, str]]:
    rows = connection.execute("SELECT id, title, destination FROM trips ORDER BY id").fetchall()
    return [
        (int(row[0]), str(row[1] or ""), str(row[2] or ""))
        for row in rows
        if not is_fixture(row[1], row[2])
    ]


def _upload_paths(connection: sqlite3.Connection, trip_ids: list[int]) -> list[Path]:
    if not trip_ids:
        return []
    columns = {column[1] for column in connection.execute("PRAGMA table_info(trip_files)")}
    if "file_path" not in columns:
        return []
    placeholders = ",".join("?" * len(trip_ids))
    rows = connection.execute(
        f"SELECT file_path FROM trip_files WHERE trip_id IN ({placeholders})", trip_ids
    ).fetchall()
    return [PROJECT_ROOT / str(row[0]) for row in rows if row[0]]


def delete_fixture_trips(connection: sqlite3.Connection, trip_ids: list[int]) -> dict[str, int]:
    placeholders = ",".join("?" * len(trip_ids))
    removed: dict[str, int] = {}
    connection.execute("BEGIN")
    for table in CHILD_TABLES:
        before = connection.total_changes
        connection.execute(f"DELETE FROM {table} WHERE trip_id IN ({placeholders})", trip_ids)
        removed[table] = connection.total_changes - before
    before = connection.total_changes
    connection.execute(f"DELETE FROM trips WHERE id IN ({placeholders})", trip_ids)
    removed["trips"] = connection.total_changes - before
    connection.commit()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually delete instead of reporting")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="path to travel_agent.db")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"database not found: {args.db}")
        return 1

    connection = sqlite3.connect(args.db)
    try:
        fixtures = find_fixture_trips(connection)
        keep = keep_trips(connection)
        print(f"total trips      : {len(fixtures) + len(keep)}")
        print(f"fixture trips    : {len(fixtures)}  (would be deleted)")
        print(f"trips to keep    : {len(keep)}")
        for trip_id, title, destination in keep:
            print(f"    keep #{trip_id} {title} ({destination})")

        if not args.apply:
            print("\nreport only. Re-run with --apply to delete the fixture trips.")
            return 0

        uploads = _upload_paths(connection, [trip_id for trip_id, _, _ in fixtures])
        delete_fixture_trips(connection, [trip_id for trip_id, _, _ in fixtures])

        removed_files = 0
        for path in uploads:
            if path.is_file() and path.is_relative_to(PROJECT_ROOT / "data" / "uploads"):
                path.unlink()
                removed_files += 1
        print(f"\ndeleted {len(fixtures)} fixture trips and {removed_files} uploaded files")
        print(f"trips remaining  : {connection.execute('SELECT COUNT(*) FROM trips').fetchone()[0]}")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
