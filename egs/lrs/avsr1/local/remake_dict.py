import json
import sys

def main(input_file, output_file, nbpe):
    with open(input_file, "r") as read_file:
        data = json.load(read_file)
        dicts = {value: key for key, value in data['char_list_dict'].items()}
        with open(output_file, "w") as output:
            output.writelines([dicts[i] + " " + str(i) + "\n" for i in range(1,int(nbpe) + 1)])

main(sys.argv[1],sys.argv[2],sys.argv[3])
