# Define the path to the original file and the new file
original_file_path = "exp/lm_stats_en_bpe5000/train/text_shape.bpe"
filtered_file_path = "exp/lm_stats_en_bpe5000/train/text_shape_new.bpe"


# Function to filter lines and write to a new file
def filter_and_write_lines(original_file, new_file):
    with open(original_file, "r") as infile, open(new_file, "w") as outfile:
        for line in infile:
            # Split the line and get the last element, assuming it's a number
            parts = line.strip().split()
            if parts:  # Check if the line is not empty
                number = int(parts[-1].split(",")[0])
                # Write the line to the new file if the number is <= 500
                if number <= 500:
                    outfile.write(line)
                else:
                    print(f"Line with number {number} was not written to the new file")


# Call the function with the paths to the original and new files
filter_and_write_lines(original_file_path, filtered_file_path)

# This code assumes that the original file 'original_file.txt' is already filled
# with the provided content. You'll need to create this file and populate it accordingly.
