"""
Based on https://github.com/monirome/AphasiaBank/blob/main/clean_transcriptions.ipynb
"""
import json
import os
from argparse import ArgumentParser

import pylangacq as pla


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--transcript-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # get a list of all speakers
    files = []
    for file in os.listdir(args.transcript_dir):
        if os.path.isfile(os.path.join(args.transcript_dir, file)) and file.endswith(
            ".cha"
        ):
            files.append(file)
    print(f"{len(files)} speakers in total")

    spk_ids = []
    spk2aphasia_type = {}
    spk2aphasia_severity = {}
    # all_aphasia_types = set()
    # '', 'notaphasicbywab', 'conduction', 'anomic', 'global', 'transsensory', 'transmotor', 'broca', 'aphasia', 'wernicke', 'control'
    with open(os.path.join(out_dir, "spk_info.txt"), "w") as f:
        f.write("spk\twab_aq\tseverity\taphasia_type\n")
        for file in files:
            spk = file.split(".cha")[0]
            assert spk not in spk_ids, spk  # spk is unique id for participants
            spk_ids.append(spk)

            path = os.path.join(args.transcript_dir, file)
            chat: pla.Reader = pla.read_chat(path)

            header = chat.headers()
            # sex = header[0]['Participants']['PAR']['sex']  # sex information
            # age = header[0]['Participants']['PAR']['age']  # age information, format year;month.day
            # age = int(age.split(';')[0])
            aphasia_type = header[0]["Participants"]["PAR"]["group"].lower()

            wab_aq = header[0]["Participants"]["PAR"]["custom"]  # WAB_AQ information
            if wab_aq == "":
                # print(f'Cannot find the WAB_AQ of {spk}')
                wab_aq = "none"
                severity = "none"
            else:
                wab_aq = float(wab_aq)

                if 0 < wab_aq <= 25:
                    severity = "very_severe"
                elif 25 < wab_aq <= 50:
                    severity = "severe"
                elif 50 < wab_aq <= 75:
                    severity = "moderate"
                else:
                    severity = "mild"

            # all_aphasia_types.add(aphasia_type)
            f.write(f"{spk}\t{wab_aq}\t{severity}\t{aphasia_type}\n")

            spk2aphasia_type[spk] = aphasia_type
            spk2aphasia_severity[spk] = severity

    # print("All aphasia types:", all_aphasia_types)

    with open(os.path.join(out_dir, "spk2aphasia_type"), "w") as f:
        json.dump(spk2aphasia_type, f)
    with open(os.path.join(out_dir, "spk2aphasia_severity"), "w") as f:
        json.dump(spk2aphasia_severity, f)


if __name__ == "__main__":
    main()
