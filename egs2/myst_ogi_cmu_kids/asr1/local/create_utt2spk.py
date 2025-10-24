# File: create_utt2spk.py

for split in ["train", "dev", "test"]:
    input_file = "data_ogi_spon/" + split + "/utt.list"
    output_file = "data_ogi_spon/" + split + "/utt2spk"

    # Open the input file and the output file
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            utt_id = line.strip()  # Remove any trailing whitespace or newline
            spk_id = utt_id[:5]  # Extract the first 5 characters as the speaker ID
            outfile.write(f"{utt_id} {spk_id}\n")  # Write utt_id and spk_id to the
