import re

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

# Example usage:
file_write=open("exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_complete/decode_asr_topk_20epoch/codec_ssl_cot_full_utt2spk_eval2000_response_combined_asr/text_new","w")
for generated_text in open("exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_complete/decode_asr_topk_20epoch/codec_ssl_cot_full_utt2spk_eval2000_response_combined_asr/text"):
    filtered_text = detect_repetition(generated_text.strip())
    # if "." in filtered_text:
    #     filtered_text=".".join(filtered_text.split(".")[:-1])+"."
    file_write.write(filtered_text+"\n")
print(filtered_text)