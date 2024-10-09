## j 68 j 68 j 68 j 68 <svs_placeholder> i 68 i 68 i 68 i 68 i 68 i 68 i 68 i 68 i 68 i 68   n 70 n 70   ian 70 ian 70 ian 70 ian 70 ian 70 ian 70 ian 70 ian 70   q 68 q 68 q 68 q 68   ian 68 ian 68 ian 68 ian 68 ian 68 ian 68 ian 68 ian 68 ian 68 ian 68 ian 68   d 66 d 66   e 66 e 66 e 66 e 66   h 66 h 66 h 66 h 66   uan 66 uan 66 uan 66 uan 66 uan 66 uan 66 uan 66 uan 66 uan 66   x 63 x 63 x 63 x 63 x 63 x 63 x 63   iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63   j 68 j 68 j 68 j 68   i 68 i 68 i 68 i 68 i 68 i 68 i 68 i 68   n 70 n 70   ian 70 ian 70 ian 70 ian 70 ian 70 ian 70 ian 70 ian 70   h 68 h 68 h 68 h 68 h 68   ou 68 ou 68 ou 68 ou 68 ou 68 ou 68 ou 68 ou 68 ou 68 ou 68 ou 68   d 66 d 66   e 66 e 66 e 66 e 66   y 66 y 66 y 66 y 66   van 66 van 66 van 66 van 66 van 66 van 66 van 66 van 66 van 66   l 63 l 63 l 63 l 63 l 63 l 63   iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63 iang 63   AP 0 AP 0 AP 0 AP 0 AP 0 AP 0 AP 0 AP 0 AP 0 AP 0 AP 0
## already tokenized in label
## find all the identical tokens in label

import os
import argparse

def dump_svs_tokenls(input, output):
    print(f"Collecting svs token list from label.")
    token_set = set()
    with open(input, 'r') as infile:
        lines = infile.readlines() # all lines
    for line in lines:
        ele_ls = line.split(" ")
        for eid in range(1,len(ele_ls)): # cut filename
            ele = ele_ls[eid].rstrip("\n")   # cut '\n'
            token_set.add(ele)

    print(f"svs token set {token_set}")

    with open(output, 'w') as outfile:
        for token in token_set:
            outfile.write(token + '\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="""
        generate svs token list from label file""")
    parser.add_argument("--input", type = str, required = True,
                        help="input label path")
    parser.add_argument("--output", type = str, required = True,
                        help="output token list")
    args = parser.parse_args()

    dump_svs_tokenls(args.input, args.output)