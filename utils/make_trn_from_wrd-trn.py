import os 
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)

    args = parser.parse_args()

    input_file = open(args.input_path, 'r', encoding='utf-8')
    output_file = open(args.output_path, 'w', encoding='utf-8')
    for input_line in input_file.readlines():
        trigger = 0
        output_line = ''
        for i in range(len(input_line)):
            if input_line[i] == '(':
                trigger = 1
            if trigger == 0:
                output_line += (input_line[i] + ' ')
            else:
                output_line += input_line[i]
        output_file.write(output_line)
