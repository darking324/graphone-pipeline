#!/usr/bin/env python3
"""Create the public Google Sheet for the GraphOne demo task.

Usage:
    python scripts/create_sheet.py [--title "..."]

Reads the 6 CSVs from data/output/, creates a spreadsheet, adds one tab
per dataset, writes the values, and shares it publicly (anyone with the
link can view). Prints the Sheet URL on success.

REQUIRES the Google Sheets + Google Drive connectors to be connected.
If they are not connected this script exits with code 2 and prints the
connect links — connect them, then re-run this script (it is idempotent
in the sense that it always creates a fresh spreadsheet; delete any
earlier partial one from Drive if needed).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "data", "output")
GWS = "/opt/hatch/bin/hatch_gws_cli"

TABS = [
    ("Startups", "startups.csv"),
    ("Products", "products.csv"),
    ("Research Papers", "papers.csv"),
    ("Jobs", "jobs.csv"),
    ("News", "news.csv"),
    ("Entity Mapping Log", "entity_mapping.csv"),
]

SHEETS_CONNECT = "https://agent.meta.ai/connectors/connect/google_sheets"
DRIVE_CONNECT = "https://agent.meta.ai/connectors/connect/google_drive"


def run_gws(args: list[str], *, expect_ok: bool = True) -> dict:
    proc = subprocess.run(
        [GWS, *args], capture_output=True, text=True, timeout=300
    )
    out = proc.stdout.strip()
    combined = out + "\n" + proc.stderr
    if "missing access_token surrogate" in combined or "not_connected" in combined:
        print("ERROR: Google Sheets/Drive is not connected.", file=sys.stderr)
        print(f"Connect Google Sheets: {SHEETS_CONNECT}", file=sys.stderr)
        print(f"Connect Google Drive:  {DRIVE_CONNECT}", file=sys.stderr)
        print("Then re-run: python scripts/create_sheet.py", file=sys.stderr)
        sys.exit(2)
    if proc.returncode != 0:
        print(f"ERROR: gws {' '.join(args[:3])} failed:\n{out}\n{proc.stderr}",
              file=sys.stderr)
        if expect_ok:
            sys.exit(1)
        return {}
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        return {"raw": out}


def read_csv_rows(path: str) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [list(row) for row in csv.reader(f)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--title",
        default="GraphOne / FrontierAtlas — AI Intelligence Graph (Demo Task)",
    )
    args = ap.parse_args()

    for _, filename in TABS:
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(path):
            print(f"ERROR: missing {path} — run the pipeline first.",
                  file=sys.stderr)
            return 1

    print("Creating spreadsheet...")
    created = run_gws([
        "sheets", "spreadsheets", "create",
        "--json", json.dumps({"properties": {"title": args.title}}),
    ])
    spreadsheet_id = created.get("spreadsheetId")
    if not spreadsheet_id:
        print(f"ERROR: unexpected create response: {created}", file=sys.stderr)
        return 1
    print(f"Spreadsheet ID: {spreadsheet_id}")

    print("Adding tabs...")
    run_gws([
        "sheets", "spreadsheets", "batchUpdate",
        "--params", json.dumps({"spreadsheetId": spreadsheet_id}),
        "--json", json.dumps({"requests": [
            {"addSheet": {"properties": {"title": name}}} for name, _ in TABS
        ]}),
    ])

    for name, filename in TABS:
        rows = read_csv_rows(os.path.join(OUTPUT_DIR, filename))
        print(f"Writing {name}: {len(rows)-1} data rows...")
        # values.update in chunks of 500 rows (header + data)
        for start in range(0, len(rows), 500):
            chunk = rows[start:start + 500]
            range_a1 = f"'{name}'!A{start + 1}"
            run_gws([
                "sheets", "spreadsheets", "values", "update",
                "--params", json.dumps({
                    "spreadsheetId": spreadsheet_id,
                    "range": range_a1,
                    "valueInputOption": "RAW",
                }),
                "--json", json.dumps({"values": chunk}),
            ])

    print("Sharing publicly (anyone with the link can view)...")
    run_gws([
        "drive", "permissions", "create",
        "--params", json.dumps({"fileId": spreadsheet_id}),
        "--json", json.dumps({"role": "reader", "type": "anyone"}),
    ])

    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    print(f"\nPUBLIC GOOGLE SHEET: {url}")
    with open(os.path.join(ROOT, "data", "sheet_url.txt"), "w") as f:
        f.write(url + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
