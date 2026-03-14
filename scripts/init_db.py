#!/usr/bin/env python3
"""
Database initialiser for Polymarket Tail Risk Hunter.

Creates the SQLite database and applies all schema migrations.
Safe to run multiple times (idempotent — uses CREATE TABLE IF NOT EXISTS).

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --db-path /custom/path/polybot.db
"""
import argparse
import os
import sys

# Allow running from repo root or from scripts/ directory
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from modules.portfolio_tracker import Database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the Polymarket bot database")
    parser.add_argument(
        "--db-path",
        default=os.environ.get("DB_PATH", "data/polybot.db"),
        help="Path to SQLite database file (default: data/polybot.db)",
    )
    args = parser.parse_args()

    print(f"Initialising database at: {args.db_path}")
    try:
        db = Database(db_path=args.db_path)
        # Verify tables exist
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables present: {', '.join(tables)}")

        # Verify indexes
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        print(f"Indexes present: {', '.join(indexes)}")

        print(f"\nDatabase ready at: {args.db_path}")
        print("WAL mode: enabled")
        print("Busy timeout: 5000ms")
        return 0
    except Exception as exc:
        print(f"ERROR: Failed to initialise database: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
