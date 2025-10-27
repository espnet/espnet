import re
import sys


def detect_repetition(text, threshold=3):
    """
    Detects excessive repetition in a text and truncates it if needed.
    
    Args:
        text (str): The generated output text.
        threshold (int): Number of times a phrase can repeat before cutting off.
    
    Returns:
        str: Truncated text if repetition is detected.
    """
    words = text.split()
    n = len(words)

    for window_size in range(2, min(20, n // 2)):  # Check for repeated phrases of increasing length
        seen_phrases = {}
        for i in range(n - window_size):
            phrase = " ".join(words[i:i + window_size])
            if phrase in seen_phrases:
                seen_phrases[phrase] += 1
                if seen_phrases[phrase] >= threshold:
                    return " ".join(words[:i])  # Cut off before repetition starts
            else:
                seen_phrases[phrase] = 1

    return text  # Return original text if no excessive repetition is found


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(output_file, "w") as file_write:
        with open(input_file, "r") as file_read:
            for generated_text in file_read:
                filtered_text = detect_repetition(generated_text.strip())
                file_write.write(filtered_text + "\n")
    
    print(f"Processing complete. Output written to {output_file}")