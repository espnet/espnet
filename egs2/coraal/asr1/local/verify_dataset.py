import glob
import os.path
import sys

import pandas as pd

# go through the metadata and check that all the files are there
if len(sys.argv) != 2:
    print("Usage: python verify_dataset.py <data_path>")
    sys.exit(1)
data_path = sys.argv[1]

expected_count, missing_count = 0, 0
for metadata_file in glob.glob(f"{data_path}/*_metadata*.txt"):
    df = pd.read_csv(metadata_file, sep="\t")
    for file in df["CORAAL.File"]:
        if file == "VLD_se0_ag2_f_01_2":
            # the only missing file
            continue
        expected_count += 1
        if not os.path.isfile(f"{data_path}/{file}.wav"):
            print(file + ".wav is missing")
            missing_count += 1

print("Expected:", expected_count)
print("Missing:", missing_count)

if expected_count == 0 or missing_count > 0:
    sys.exit(1)
