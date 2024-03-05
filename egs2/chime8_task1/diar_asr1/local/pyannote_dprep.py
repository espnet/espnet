import argparse
import glob
import json
import math
import os.path
import re
from pathlib import Path

import soundfile as sf
import tqdm
from pyannote.metrics.segmentation import Annotation, Segment


def round_down(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def json2annotation(chimelike_jsonf, uri=None, spk_prefix=None):
    with open(chimelike_jsonf, "r") as f:
        json_ann = json.load(f)

    def to_annotation(segs, uri=None, spk_prefix=None):
        out = Annotation(uri=uri)
        for s in segs:
            speaker = s["speaker"]
            if spk_prefix is not None:
                speaker = spk_prefix + "_" + speaker
            start = float(s["start_time"])
            end = float(s["end_time"])
            out[Segment(start, end)] = speaker
        return out

    return to_annotation(json_ann, uri, spk_prefix)


# FIXME add notsofar1 synth and AMI in the fine-tuning
def prepare4pyannote(
    chime7dasr_root, target_dir="./data/pyannote_diarization", falign_annotation=None
):
    for dset in ["chime6", "dipco", "mixer6"]:
        print("Running Pyannote data preparation for {} scenario".format(dset))
        c_scenario = os.path.join(chime7dasr_root, dset)
        c_splits = ["train", "dev"] if dset == "chime6" else ["dev"]
        # exclude close-talk CH01, CH02, CH03 and P[0-9]+
        if dset in ["chime6", "dipco"]:
            mic_regex = "(U[0-9]+)"
            sess_regex = "(S[0-9]+)"
        else:
            mic_regex = "(?!CH01|CH02|CH03)(CH[0-9]+)"
            sess_regex = "([0-9]+_[0-9]+_(LDC|HRM)_[0-9]+)"

        for split in c_splits:
            audio_files = glob.glob(os.path.join(c_scenario, "audio", split, "*.wav"))
            audio_files += glob.glob(os.path.join(c_scenario, "audio", split, "*.flac"))
            audio_files = [x for x in audio_files if re.search(mic_regex, Path(x).stem)]
            # exclude close-talk ones here
            if falign_annotation is not None and dset == "chime6":
                json_folder = os.path.join(falign_annotation, split, "*.json")
            else:
                json_folder = os.path.join(
                    c_scenario, "transcriptions_scoring", split, "*.json"
                )

            json_annotations = glob.glob(json_folder)
            sess2json = {}
            for json_f in json_annotations:
                sess2json[Path(json_f).stem] = json2annotation(
                    json_f, uri="PLACEHOLDER", spk_prefix=dset
                )

            uem = os.path.join(c_scenario, "uem", split, "all.uem")
            with open(uem, "r") as f:
                uem_lines = f.readlines()
            sess2uem = {}
            for x in uem_lines:
                sess_id, _, start, stop = x.rstrip("\n").split(" ")
                sess2uem[sess_id] = (float(start), float(stop))

            # now for each recording uri we need an rttm and an uem and dump it
            # into the target dir
            c_rttm_dir = os.path.join(target_dir, dset, split, "rttm")
            Path(c_rttm_dir).mkdir(exist_ok=True, parents=True)
            c_uem_dir = os.path.join(target_dir, dset, split, "uem")
            Path(c_uem_dir).mkdir(exist_ok=True, parents=True)
            c_uri_dir = os.path.join(target_dir, dset, split, "uris")
            Path(c_uri_dir).mkdir(exist_ok=True, parents=True)

            to_uri_list = []
            for audio_f in tqdm.tqdm(audio_files):
                filename = Path(audio_f).stem
                to_uri_list.append(filename)  # append uri here
                session = re.search(sess_regex, filename).group()  # sess regex here
                c_ann = sess2json[session]
                c_uem = sess2uem[session]
                # put the filename here
                with open(os.path.join(c_rttm_dir, filename + ".rttm"), "w") as f:
                    f.writelines(c_ann.to_rttm().replace("PLACEHOLDER", filename))
                # writing uem
                # NOTE: we recreate here uem for each file, to assure that they
                # will be not out of bounds as pyannote otherwise will throw errors
                # during training.
                c_uem = [
                    c_uem[0],
                    round_down(
                        sf.SoundFile(audio_f).frames / sf.SoundFile(audio_f).samplerate,
                        2,
                    ),
                ]
                # in evaluation we use the correct CHiME-7 DASR .uem
                with open(os.path.join(c_uem_dir, filename + ".uem"), "w") as f:
                    f.write("{} 1 {} {}\n".format(filename, c_uem[0], c_uem[1]))

            # write the files uris
            to_uri_list = sorted(to_uri_list)
            to_uri_list = [x + "\n" for x in to_uri_list]
            with open(os.path.join(c_uri_dir, "uri.txt"), "w") as f:
                f.writelines(to_uri_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Preparing CHiME-7 DASR data for fine-tuning"
        "the Pyannote segmentation model.",
        add_help=True,
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "-r,--dasr_root",
        type=str,
        default="dev",
        help="Folder containing the main folder of CHiME-7 DASR dataset.",
        metavar="STR",
        dest="dasr_root",
    )
    parser.add_argument(
        "--output_root",
        required=False,
        default="./data/pyannote_diarization",
        type=str,
        metavar="STR",
        dest="output_root",
        help="Path where the Pyannote data manifests files are created."
        "Note that these should match the entries in the database.yml file in "
        "this folder.",
    )
    parser.add_argument(
        "--falign_dir",
        type=str,
        required=False,
        default="",
        metavar="STR",
        dest="falign_dir",
        help="Path to the CHiME-7 DASR CHiME-6 JSON annotation obtained using "
        "forced alignment. This is optional. "
        "The forced alignment annotation is available in "
        "https://github.com/chimechallenge/CHiME7_DASR_falign",
    )
    args = parser.parse_args()

    prepare4pyannote(
        args.dasr_root,
        args.output_root,
        None if not len(args.falign_dir) else args.falign_dir,
    )
