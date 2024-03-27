# make_subprotocols.py

# Makes sub protocols from the ASVspoof protocol and outputs them in Vox1-O style

import argparse


def process_trials(input_file, output_file, trial_type):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split()
            scenario = " ".join(parts[2:])
            if trial_type == "SV":
                label = (
                    "1"
                    if scenario == "bonafide target"
                    else "0" if scenario == "bonafide nontarget" else None
                )
            else:
                match_scenario = f"{trial_type} spoof"
                label = (
                    "1"
                    if scenario == "bonafide target"
                    else "0" if scenario == match_scenario else None
                )

            if label is not None:
                outfile.write(f"{label} {parts[0]} {parts[1]}\n")


def main():
    parser = argparse.ArgumentParser(description="Make subprotocol files")
    parser.add_argument("--in_file", required=True, help="Input file with trials")
    parser.add_argument("--out_file", required=True, help="Output file name")
    parser.add_argument(
        "--trial_type",
        required=True,
        choices=["SV"] + [f"A{i:02d}" for i in range(7, 20)],
        help="Type of trial to filter for",
    )

    args = parser.parse_args()

    process_trials(args.in_file, args.out_file, args.trial_type)


if __name__ == "__main__":
    main()
