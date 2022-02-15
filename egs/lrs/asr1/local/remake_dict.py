import json
import sys

def main(input_file, output_file):
    with open(input_file, "r") as read_file:
        data = json.load(read_file)
        dicts = {value: key for key, value in data['char_list_dict'].items()}
        with open(output_file, "w") as output:
            output.writelines([dicts[i] + " " + str(i) + "\n" for i in range(1,501)])

# hand over parameter overview
# sys.argv[1] = input_file (str): The directory from which the dict should be used, 
#               e.g. Pretrained LM under exp/pretrainedlm/model.json ${dict}
# sys.argv[2] = output_file: Replaced dictionary 

main(sys.argv[1],sys.argv[2])
