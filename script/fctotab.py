import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def readable_time(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def member_columns(record):
    return [
        key
        for key in record
        if "temperature_2m" in key and isinstance(record[key], list)
    ]


def output_path(input_path, output_dir, line_number):
    name = f"{input_path.stem}_{line_number}.csv"
    return output_dir / name


def convert_record(record, destination):
    times = record.get("time")
    if not isinstance(times, list):
        return

    columns = member_columns(record)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("time", *columns))
        writer.writerows(
            (readable_time(timestamp), *(record[column][index] for column in columns))
            for index, timestamp in enumerate(times)
        )


def convert_file(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                convert_record(
                    json.loads(line),
                    output_path(input_path, output_dir, line_number),
                )


def main():
    parser = argparse.ArgumentParser(
        description="Convert fc.jsonl records to per-record CSV files."
    )
    parser.add_argument("input", type=Path, help="fc.jsonl file to convert")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("test-data"),
        help="directory for generated CSV files (default: test-data)",
    )
    args = parser.parse_args()
    convert_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()
