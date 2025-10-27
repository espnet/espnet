import re
import sys


def truncate_by_length_ratio(generated_text, src_text, threshold=3.0):
    """
    Truncates generated text if it exceeds a length ratio threshold compared to source text.
    
    Args:
        generated_text (str): The generated output text.
        src_text (str): The source reference text.
        threshold (float): Maximum allowed length ratio (default: 3.0).
    
    Returns:
        str: Truncated text if ratio exceeds threshold.
    """
    gen_words = generated_text.split()[1:]  # Skip first word (likely ID)
    src_words = src_text.split()[1:]  # Skip first word (likely ID)
    
    a1 = len(gen_words)
    a2 = len(src_words)
    
    if a1 / a2 > threshold:
        # Reconstruct with first word (ID)
        first_word = generated_text.split()[0]
        generated_text_content = " ".join(gen_words)
        
        if "." in generated_text_content:
            found = True
            while a1 / a2 > threshold:
                if "." not in generated_text_content:
                    generated_text_content = " ".join(generated_text_content.split()[:int(threshold * a2)])
                    found = False
                    break
                generated_text_content = ".".join(generated_text_content.split(".")[:-1])
                a1 = len(generated_text_content.split())
            
            if found:
                generated_text_content = generated_text_content + "."
        else:
            generated_text_content = " ".join(generated_text_content.split()[:int(threshold * a2)])
        
        return first_word + " " + generated_text_content
    
    return generated_text


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <source_file> <output_file>")
        print("  input_file: Generated text file to process")
        print("  source_file: Source reference text file")
        print("  output_file: Output file for processed text")
        sys.exit(1)
    
    input_file = sys.argv[1]
    source_file = sys.argv[2]
    output_file = sys.argv[3]
    threshold = 3.0
    
    # Read source file into array
    with open(source_file, "r") as src_file:
        src_arr = [line for line in src_file]
    
    # Process generated text
    with open(output_file, "w") as file_write:
        with open(input_file, "r") as file_read:
            count = 0
            for generated_text in file_read:
                filtered_text = truncate_by_length_ratio(
                    generated_text.strip(), 
                    src_arr[count].strip(), 
                    threshold
                )
                file_write.write(filtered_text.strip() + "\n")
                count += 1
    
    print(f"Processing complete. Processed {count} lines with threshold={threshold}")
    print(f"Output written to {output_file}")