import os
import json
import argparse
# score.scp(json) --> text

def process_scp_file(output_dir):
    scp_file = os.path.join(output_dir, 'score.scp')
    output_str = ""
    
    with open(scp_file, 'r') as f:
        for line in f:
            key, json_path = line.strip().split()
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            
            tempo = data['tempo']
            note_list = data['note']

            file_output_str = f"{tempo} "
            for note in note_list:
                file_output_str += " ".join(map(str, note)) + " "
            
            output_str += file_output_str.strip() + "\n"
    
    output_file = os.path.join(output_dir, 'score')
    with open(output_file, 'w') as f:
        f.write(output_str)
    
    print(f"Successfully converted score format of {output_dir.split('/')[-1]}.")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="""
    Convert score file format.""")
    parser.add_argument("--data_folder", type = str, required = True,
                        help="in processing dataset")
    args = parser.parse_args()
    process_scp_file(args.data_folder)