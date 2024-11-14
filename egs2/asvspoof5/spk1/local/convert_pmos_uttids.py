# convert_pmos_uttids.py
# given an input utt2pmos file with columns: uttid1 pmos
# and a wav.scp file with columns: uttid2 path
# and a utt2spk file with columns: uttid2 pmos, therefore 
# getting the global uttid

# This script will convert the uttids in the utt2pmos file to the global uttid

import argparse
import os
import sys
from tqdm import tqdm

def main(args):
    # Read input files
    with open(args.in_utt2pmos, "r") as f:
        lines_utt2pmos = f.readlines()
    with open(args.wavscp, "r") as f:
        lines_wavscp = f.readlines()

    print(f"Read {len(lines_utt2pmos)} lines from utt2pmos")
    print(f"Read {len(lines_wavscp)} lines from wavscp")

    # Create wavscp dictionary
    wavscp_dict = {}
    suffix_lookup = {}
    # prefix is T_ for the short ID format if it in the 
    for line in lines_wavscp:
        utt_id, path = line.strip().split(" ")
        wavscp_dict[utt_id] = path
        # Extract the part after the last underscore as the suffix
        suffix, set = utt_id.split('_')[-1], utt_id.split('_')[-2]
        set = set[-1]
        suffix_lookup[set + '_' + suffix] = utt_id
        
    print(f"Created lookup with {len(suffix_lookup)} entries")
    if len(suffix_lookup) > 0:
        print("Sample suffix_lookup entry:", list(suffix_lookup.items())[0])

    # Count matches for debugging
    match_count = 0
    
    # Write matches using direct suffix lookup
    with open(os.path.join(args.out, "utt2pmos"), "w") as f:
        for line in tqdm(lines_utt2pmos):
            short_utt_id, pmos = line.strip().split(" ")
            if short_utt_id in suffix_lookup:
                full_id = suffix_lookup[short_utt_id]
                f.write(f"{full_id} {pmos}\n")
                match_count += 1

    print(f"Total matches found: {match_count}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts uttids in utt2pmos file to global uttid"
    )
    parser.add_argument(
        "--in_utt2pmos", type=str, help="path to the input utt2pmos file"
    )
    parser.add_argument(
        "--wavscp", type=str, help="path to wav.scp file"
    )
    parser.add_argument(
        "--out", type=str, help="output directory"
    )
    args = parser.parse_args()
    main(args)