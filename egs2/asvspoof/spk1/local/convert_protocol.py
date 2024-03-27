# convert_protocol.py

# Converts ASVspoof protocol to Vox1-O style
import argparse


def process_trials(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split()
            scenario = " ".join(parts[2:])
            label = "1" if scenario.lower() == "bonafide target" else "0"
            outfile.write(f"{label} {parts[0]} {parts[1]}\n")


def main():
    parser = argparse.ArgumentParser(description="Make protocol Vox1-O style")
    parser.add_argument("--in_file", required=True, help="Input file w trials")
    parser.add_argument("--out_file", required=True, help="Output")

    args = parser.parse_args()

    process_trials(args.in_file, args.out_file)


if __name__ == "__main__":
    main()
