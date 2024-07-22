#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import re
import string
import copy
from zhon.hanzi import punctuation as zh_punctuation
all_punctuation = string.punctuation + zh_punctuation

# Only support 9 languages
lang_map = {
    "eng": None,
    "zho": None,
    "fra": None,
    "deu": None,
    "ita": None,
    "spa": None,
    "nld": None,
    "por": None,
    "pol": None,
}

# Thank you! GPT!
def compute_edit_distance_list_detailed(s1, s2, track_back=True):
    # Initialize the table with 0's.
    dp = [[0 for x in range(len(s2) + 1)] for x in range(len(s1) + 1)]

    # Fill the table in a bottom-up manner.
    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i == 0:
                dp[i][j] = j  # Minimum operations = j
            elif j == 0:
                dp[i][j] = i  # Minimum operations = i
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],    # Insert
                                   dp[i-1][j],    # Remove
                                   dp[i-1][j-1])  # Replace
    
    if not track_back:
        return dp[len(s1)][len(s2)]

    # Initialize variables for tracking changes.
    substitutions, insertions, deletions = [], [], []

    # Backtrack to find the sequence of operations.
    i, j = len(s1), len(s2)
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            i, j = i-1, j-1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            substitutions.append((i-1, (s1[i-1], s2[j-1])))
            i, j = i-1, j-1
        elif dp[i][j] == dp[i][j-1] + 1:
            insertions.append((j-1, s2[j-1]))
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            deletions.append((i-1, s1[i-1]))
            i -= 1

    # Check for any remaining operations in s1 or s2.
    while i > 0:
        deletions.append((i-1, s1[i-1]))
        i -= 1
    while j > 0:
        insertions.append((j-1, s2[j-1]))
        j -= 1

    # Reverse lists to maintain the original order.
    insertions.reverse()
    deletions.reverse()
    substitutions.reverse()

    return dp[len(s1)][len(s2)], substitutions, insertions, deletions

def process_one_example(data, llm_str):
    
    # (3) Get the two lists from both side.
    if llm_str.startswith('"') and llm_str.endswith('"'):
        llm_str = llm_str[1:-1]
    
    src_text = data['text'].strip().split()
    llm_text = llm_str.strip().split()
    llm_text = llm_text[:min(len(llm_text), int(len(src_text) * 1.3))] # in case it's too long
    
    # (4) Compute edit-distance and statistics
    nerrors, sub, ins, dele = compute_edit_distance_list_detailed(src_text, llm_text)
    
    # (5) Do replacement and insertions
    for idx, (s1, s2) in sub:
        if re.sub(f'[{re.escape(all_punctuation)}]', '', s2).lower() == re.sub(f'[{re.escape(all_punctuation)}]', '', s1).lower():
            src_text[idx] = s2
    
    for idx, s2 in ins:
        if s2 in all_punctuation:
            src_text.insert(idx, s2)
    
    # (6) Check if too many errors:
    nerrors, sub, ins, dele = compute_edit_distance_list_detailed(src_text, llm_text)
    if nerrors > len(src_text) * 0.3:
        print(f'too many errors detected: {src_text} {llm_text}', nerrors, flush=True)
        print('sub', sub)
        print('ins: ', ins)
        print('dell: ', dele)
        return False, "too many errors"

    # (7) Recover the format:

    data['text'] = " ".join(src_text)

    return True, None


def main(input_file, llm_file, output_file):
    logging.basicConfig(
        level="INFO",
        format=f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Parse input:
    input_data = {}
    for line in open(input_file, encoding='utf-8'):
        uttid, content = line.strip().split(maxsplit=1)
        input_data[uttid] = {'text': content}
    
    # Parse LLM output:
    llm_data = {}
    for line in open(llm_file, encoding='utf-8'):
        uttid, content = line.strip().split(maxsplit=1)
        if uttid in input_data:
            llm_data[uttid] = content

    succ_count = 0
    for k, v in llm_data.items():
        prev = copy.deepcopy(input_data[k])
        succ, info = process_one_example(input_data[k], llm_data[k])
        if succ:
            succ_count += 1
        else:
            logging.info(f"{k} fails: {info}")
    logging.info(f"Finally, {len(llm_data)} to process. {succ_count} success")

    # Write the output file
    writer = open(output_file, 'w', encoding='utf-8')
    for k, v in input_data.items():
        writer.write(f'{k} {v["text"]}\n')


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some files.")

    # Add the arguments
    parser.add_argument('--input_file', '-i', type=str, help='The path to the input file')
    parser.add_argument('--llm_file', '-l', type=str, help='The path to the LLM output file')
    parser.add_argument('--output_file', '-o', type=str, help='The path to the output file')

    # Execute the parse_args() method
    args = parser.parse_args()

    main(args.input_file, args.llm_file, args.output_file)
