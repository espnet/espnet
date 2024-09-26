import os
import argparse

def add_phn_duration(data_folder):
    score_file = os.path.join(args.data_folder, "score")
    label_path = os.path.join(args.data_folder, "label")
    out_file = os.path.join(args.data_folder, "new_label")
            
    midi_ls = [] # 2d list
    with open(score_file, 'r') as score_file:
        for line in score_file:
            line_midi = []
            parts = line.split()
            tempo = parts[0]
            data_unit = parts[1:]
            
            for i in range(0, len(data_unit), 3):
                unit_len = len(data_unit[i+2].split('_')) # num of phns
                for j in range(unit_len):
                    line_midi.append(data_unit[i+1]) # phn, midi
            midi_ls.append(line_midi)

    with open(label_path, 'r') as label_file:
        for line in label_file:
            parts = line.strip().split()
            audio_id = parts[0]
            data_unit = parts[1:]
            
            result = [audio_id]
            
            for i in range(0, int(len(data_unit)/2)):
                phn_duration = data_unit[2*i]
                phone = data_unit[2*i+1]
                phn_midi = midi_ls[0][i]
                result.append(f"{phn_duration} {phone} {phn_midi}")
            midi_ls.pop(0)

            with open(out_file, 'a') as outfile:
                outfile.write(' '.join(result) + '\n')

    os.rename(out_file, label_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="""
    Add phn level duration from label to score""")
    parser.add_argument("--data_folder", type = str, required = True,
                        help="in processing dataset")
    args = parser.parse_args()

    add_phn_duration(args.data_folder)