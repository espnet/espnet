"""Align optional waveform references over the union of utterance keys."""

import argparse


def read_rows(path):
    """Read unique key/value rows, preserving paths and pipe commands."""
    rows = {}
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2 or parts[0] in rows:
                raise ValueError(f"Invalid or duplicate key at {path}:{line_number}")
            rows[parts[0]] = parts[1]
    return rows


def align_keys(file1, file2, output):
    """Write file2 values over the sorted union of keys, filling gaps with None."""
    data1 = read_rows(file1)
    data2 = read_rows(file2)
    with open(output, "w") as out:
        for key in sorted(data1.keys() | data2.keys()):
            out.write(f"{key} {data2.get(key, 'None')}\n")


def main():
    """Parse paths and write one aligned SCP file."""
    parser = argparse.ArgumentParser(
        description="Align file2 to the union of input keys, filling gaps with None."
    )
    parser.add_argument("file1", help="Path to the first input file")
    parser.add_argument("file2", help="Path to the values to align")
    parser.add_argument("output", help="Path to the aligned output file")
    args = parser.parse_args()
    align_keys(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
