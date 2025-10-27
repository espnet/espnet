import re
import sys
import os


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_directory>")
        print("  Example: python script.py data/train_nodup")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    input_file = os.path.join(data_dir, "text")
    output_file = os.path.join(data_dir, "text_clean")
    
    # Read input file
    with open(input_file, "r") as f:
        line_arr = [line.strip() for line in f]
    
    # Define keys to remove
    key_arr = ["[laughter]", "[noise]", "[vocalized-noise]", "[[skip]]", "[sneeze]", "[pause]"]
    
    # Process and write output
    with open(output_file, "w") as file_write:
        for line in line_arr:
            # Remove all specified keys
            for key in key_arr:
                line = line.replace(key, "")
            
            # Clean up multiple spaces
            text = re.sub(r' +', ' ', line)
            
            # Write if line has more than just ID
            if len(text.strip().split()) > 1:
                file_write.write(text.strip() + "\n")
    
    print(f"Processing complete. Cleaned text written to {output_file}")