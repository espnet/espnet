#!/usr/bin/env python3
"""Modify parquet files by replacing path prefixes."""

import argparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def load_valid_ids(filepath):
    """Load valid IDs from a file, one ID per line."""
    valid_ids = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().split()[0]
            if line:
                valid_ids.add(line)
    return valid_ids


def main():
    parser = argparse.ArgumentParser(
        description="Modify parquet files by replacing path prefixes."
    )
    parser.add_argument("--input", required=True, help="Input parquet file path")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument(
        "--original_path", default=None, help="Original path prefix to replace"
    )
    parser.add_argument(
        "--new_path", default=None, help="New path prefix to use as replacement"
    )
    parser.add_argument(
        "--valid_ids",
        default=None,
        help="File containing valid IDs (one per line). "
        "Only rows with these IDs are kept.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Number of rows to process at a time (default: 100000)",
    )
    args = parser.parse_args()

    # Load valid IDs if provided
    valid_ids = None
    if args.valid_ids is not None:
        valid_ids = load_valid_ids(args.valid_ids)
        print(f"Loaded {len(valid_ids)} valid IDs from {args.valid_ids}")

    parquet_file = pq.ParquetFile(args.input)
    schema = parquet_file.schema_arrow

    writer = None
    total_rows = 0

    kept_rows = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch], schema=schema)

            # Filter by valid IDs if provided
            if valid_ids is not None:
                id_col = table.column("utt_id")
                mask = pc.is_in(id_col, value_set=pa.array(list(valid_ids)))
                table = table.filter(mask)

            # Skip empty tables after filtering
            if table.num_rows == 0:
                total_rows += len(batch)
                print(f"Processed {total_rows} rows, kept {kept_rows}...", end="\r")
                continue

            if args.original_path is not None and args.new_path is not None:
                path_col = table.column("path")
                new_path_col = pc.replace_substring(
                    path_col,
                    pattern=args.original_path,
                    replacement=args.new_path,
                )
                col_idx = table.schema.get_field_index("path")
                table = table.set_column(col_idx, "path", new_path_col)

            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema)

            writer.write_table(table)
            total_rows += len(batch)
            kept_rows += table.num_rows
            print(f"Processed {total_rows} rows, kept {kept_rows}...", end="\r")

    finally:
        if writer is not None:
            writer.close()

    if valid_ids is not None:
        print(
            f"\nModified parquet saved to {args.output} "
            f"({kept_rows} rows kept out of {total_rows})"
        )
    else:
        print(f"\nModified parquet saved to {args.output} ({total_rows} rows)")


if __name__ == "__main__":
    main()
