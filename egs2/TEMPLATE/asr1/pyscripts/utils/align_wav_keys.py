import argparse


def align_keys(file1, file2, output):
    """Align keys from two sorted files and write aligned
    results into two separate output files."""
    # Read the files into dictionaries (keys must be unique)
    data1 = {
        line.split(None, 1)[0]: line.strip()
        for line in open(file1, "r")
        if line is not None
    }
    data2 = {
        line.split(None, 1)[0]: line.strip()
        for line in open(file2, "r")
        if line is not None
    }

    # Combine all keys
    all_keys = sorted(set(data1.keys()) | set(data2.keys()))

    # Write aligned results to the output files
    with open(output, "w") as out:
        for key in all_keys:
            if key in data2.keys():
                out.write("{}\n".format(data2[key]))
            else:
                out.write("{} None\n".format(key))

    print(f"Aligned files written to {output}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Align keys between two sorted files into separate outputs."
    )
    parser.add_argument("file1", help="Path to the first input file")
    parser.add_argument("file2", help="Path to the second input file")
    parser.add_argument("output", help="Path to the first output file")
    args = parser.parse_args()

    # Align keys and write to the output files
    align_keys(args.file1, args.file2, args.output)


if __name__ == "__main__":
    main()
