import os 
import argparse
from pypinyin import lazy_pinyin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)

    args=parser.parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as input_file, open(args.output_path, 'w', encoding='utf-8') as output_file:
        for line in input_file.readlines():
            ls = line.split(" ")
            words = ls[0]
            output_line = ''
            py_list = lazy_pinyin(words)
            for py in py_list:
                output_line += (py+' ')
            output_line += ls[1]
            output_file.write(output_line)

            

