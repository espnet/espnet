import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        print("  Example: python script.py input.txt output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Read input file
    with open(input_file, "r") as file:
        line_arr = [line for line in file]
    
    # Process and write output
    with open(output_file, "w") as file_write:
        for line in line_arr:
            if "_sample0" in line:
                file_write.write(line.replace("codec_ssl_cot_full_utt2spk_", "").replace("_sample0", ""))
    
    print(f"Processing complete. Output written to {output_file}")