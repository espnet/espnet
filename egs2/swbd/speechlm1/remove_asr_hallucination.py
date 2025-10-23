# import re

# def detect_repetition(text, threshold=5):
#     """
#     Detects excessive repetition in a text and truncates it if needed.
    
#     Args:
#         text (str): The generated output text.
#         threshold (int): Number of times a phrase can repeat before cutting off.
    
#     Returns:
#         str: Truncated text if repetition is detected.
#     """
#     words = text.split()
#     n = len(words)

#     for window_size in range(2, min(20, n // 2)):  # Check for repeated phrases of increasing length
#         seen_phrases = {}
#         for i in range(n - window_size):
#             phrase = " ".join(words[i:i + window_size])
#             if phrase in seen_phrases:
#                 seen_phrases[phrase] += 1
#                 if seen_phrases[phrase] >= threshold:
#                     return " ".join(words[:i])  # Cut off before repetition starts
#             else:
#                 seen_phrases[phrase] = 1

#     return text  # Return original text if no excessive repetition is found

# file_write=open("exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_copy/decode_asr_short_9epoch/codec_ssl_cot_full_utt2spk_test_fisher_final_nobc/src_text_new","w")
# for generated_text in open("exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_copy/decode_asr_short_9epoch/codec_ssl_cot_full_utt2spk_test_fisher_final_nobc/src_text"):
#     filtered_text = detect_repetition(generated_text.strip())
#     # if "." in filtered_text:
#     #     filtered_text=".".join(filtered_text.split(".")[:-1])+"."
#     file_write.write(filtered_text+"\n")
# print(filtered_text)

import re

# def remove_repeated_phrases(text, min_words=2, max_repeats=2):
#     """
#     Removes consecutive repeated phrases from ASR text.

#     Args:
#         text (str): ASR output text.
#         min_words (int): Minimum number of words to consider as a repeated phrase.
#         max_repeats (int): Maximum allowed repetitions before removal.

#     Returns:
#         str: Cleaned text with reduced hallucinations.
#     """
#     # Normalize spaces
#     # text = re.sub(r'\s+', ' ', text).strip()

#     words = text.split()
#     cleaned_words = []
#     phrase_counts = {}
#     for phrase_length in range(min_words, len(words)):
#         i = 0
        
#         while i < len(words)-phrase_length:
#         # Check for repeated phrases (from min_words up to the full remaining sentence)

        
#             phrase = " ".join(words[i:i + phrase_length])
#             next_phrase = " ".join(words[i + phrase_length:i + phrase_length+ phrase_length])

#             # Check if phrase has been repeated consecutively
#             if phrase==next_phrase:
#                 return " ".join(words[:i])
#             i+=1


#     return text

def remove_repeated_phrases(text, min_words=2, max_repeats=2):
    """
    Removes consecutive repeated phrases from ASR text.

    Args:
        text (str): ASR output text.
        min_words (int): Minimum number of words to consider as a repeated phrase.
        max_repeats (int): Maximum allowed repetitions before removal.

    Returns:
        str: Cleaned text with reduced hallucinations.
    """
    # Normalize spaces
    # text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    cleaned_words = []
    phrase_counts = {}
    for phrase_length in range(min_words, len(words)):
        i = 0
        
        while i < len(words)-phrase_length:
        # Check for repeated phrases (from min_words up to the full remaining sentence)

        
            phrase = " ".join(words[i:i + phrase_length])
            next_phrase = " ".join(words[i + phrase_length:i + phrase_length+ phrase_length])

            # Check if phrase has been repeated consecutively
            if phrase==next_phrase:
                if i + phrase_length+ phrase_length+ phrase_length>=len(words):
                    return " ".join(words[:i])
                else:
                   next_phrase_2 = " ".join(words[i + phrase_length+ phrase_length:i + phrase_length+ phrase_length+ phrase_length])
                   if phrase==next_phrase_2:
                    return " ".join(words[:i])
            i+=1


    return text

def remove_trailing_repetitions(text):
    """
    Removes consecutive single-word repetitions at the end of a sentence.

    Args:
        text (str): ASR output sentence.

    Returns:
        str: Cleaned sentence without trailing repetitions.
    """
    words = text.strip().split()

    # Edge case: If there's only one word or an empty string, return as is
    if len(words) <= 2:
        return text

    # Identify the last unique word before the repetition starts
    last_unique_word = words[-2]
    found=False
    for i in range(len(words) - 3, -1, -1):  # Traverse backwards
        if words[i] != last_unique_word:
            break
        last_unique_word = words[i]
        found=True

    # Remove trailing repeated words
    if found:
        cleaned_words = words[:(i+1)]
    else:
        cleaned_words = words[:i+3]

    return " ".join(cleaned_words)

# Example ASR output with repeated phrases
asr_output = """I mean, my bags have been torn apart every time I travel. I must I must have that look or something. I don't know. Mhm. I don't know. Mhm."""

cleaned_text = remove_repeated_phrases(asr_output)
print(cleaned_text)
file_write=open("/work/nvme/bbjs/arora1/speech_lm/delta_ai/espnet/egs2/swbd/speechlm1/exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_complete/decode_asr_short2_20epoch/codec_ssl_cot_full_utt2spk_eval2000_response/src_text_new","w")
for generated_text in open("/work/nvme/bbjs/arora1/speech_lm/delta_ai/espnet/egs2/swbd/speechlm1/exp/speechlm_codec_ssl_cot_full_utt2spk_librispeech_100_train_s2s_lr1e-5_fisher_complete/decode_asr_short2_20epoch/codec_ssl_cot_full_utt2spk_eval2000_response/src_text"):
    # if generated_text.split()[0]=="codec_ssl_cot_full_utt2spk_fe_03_00292-A_013206-014135_sample0":
    #     import pdb;pdb.set_trace()
    filtered_text = remove_repeated_phrases(generated_text.strip())
    filtered_text = remove_repeated_phrases(filtered_text)
    filtered_text = remove_trailing_repetitions(filtered_text)
    file_write.write(filtered_text+"\n")
print(filtered_text)