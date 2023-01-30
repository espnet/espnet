import jiwer
import os
from pathlib import Path
import glob
import json
from datetime import datetime as dt
from copy import deepcopy
from lhotse.recipes.chime6 import normalize_text_chime6, TimeFormatConverter
from jiwer.transforms import BaseRemoveTransform, RemoveKaldiNonWords
import argparse


chime6_map = {"train": ["S03", "S04", "S05", "S06", "S07", "S08", "S12", "S13", "S16", "S17",
"S18", "S22", "S23", "S24"], "dev": ["S02", "S09"],
              "eval": ["S19", "S20", "S01", "S21"]}
dipco_cv = ["S02", "S04", "S05", "S09", "S10"]
dipco_tt = ["S01", "S03", "S06", "S07", "S08"]
dipco_offset = 24

# official txt normalization for chime7 task 1 scoring,
# you are free to use whatever normalization you prefer but this
# one will be used when we score your submissions.
chime7_norm_scoring = jiwer.Compose([ #TODO
        #jiwer.SubstituteRegexes(),
        BaseRemoveTransform([c for c in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~']),
        BaseRemoveTransform(["-"]),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ToLowerCase()])

# need to remove also quotation marks and leading, trailing whitespaces and
# kaldi non-words w.r.t. lhotse one.
jiwer_chime6_scoring = jiwer.Compose([RemoveKaldiNonWords(),
                                      jiwer.RemoveEmptyStrings(),
                                      jiwer.SubstituteRegexes({r"\"": " ", "^[ \t]+|[ \t]+$": "",
                                                               r"\u2019": "'"}),
                                      jiwer.RemoveMultipleSpaces()])
# old chime-6 txt normalization for scoring
chime6_norm_scoring = lambda x : jiwer_chime6_scoring(normalize_text_chime6(x, normalize="kaldi"))

def choose_txt_normalization(scoring_txt_normalization="chime7"):
    if scoring_txt_normalization == "chime7":
        scoring_txt_normalization = chime7_norm_scoring
    elif scoring_txt_normalization == "chime6":
        scoring_txt_normalization = chime6_norm_scoring
    else:
        raise NotImplementedError("scoring text normalization should be either 'chime7' or 'chime6'")
    return scoring_txt_normalization

def prep_chime6(root_dir, out_dir,
                scoring_txt_normalization="chime7"):

    scoring_txt_normalization = choose_txt_normalization(scoring_txt_normalization)
    if Path(out_dir).exists():
        raise FileExistsError("{} appears to have been created already, "
                               "exiting. Please delete/move/rename "
                              "it if you want to re-generate CHiME7 Task 1 data".format(out_dir))
    def normalize_chime6(annotation, txt_normalizer, is_eval=False):
        annotation_scoring = []
        for ex in annotation:
            ex["start_time"] = str(TimeFormatConverter.hms_to_seconds(ex["start_time"]))
            ex["end_time"] = str(TimeFormatConverter.hms_to_seconds(ex["end_time"]))
            if is_eval and "ref" in ex.keys():
                del ex["ref"]
                del ex["location"]
                # cannot be used in inference
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    # pre-create all destination folders
    for split in ["train", "dev", "eval"]:
        Path(os.path.join(out_dir, "audio", split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, "transcriptions", split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, "transcriptions_scoring", split)).mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "eval"]:
        json_dir = os.path.join(root_dir, "transcriptions", split)
        ann_json = glob.glob(os.path.join(json_dir, "*.json"))
        assert len(ann_json) > 0, "CHiME-6 JSON annotation was not found in {}, please check if " \
                                  "CHiME-6 data was downloaded correctly and the CHiME-6 main dir " \
                                  "path is set correctly".format(json_dir)
        # we also create audio files symlinks here
        audio_files = glob.glob(os.path.join(root_dir, "audio", split, "*.wav"))
        sess2audio = {}
        for x in audio_files:
            session_name = Path(x).stem.split("_")[0]
            if session_name not in sess2audio:
                sess2audio[session_name] = [x]
            else:
                sess2audio[session_name].append(x)

        # for each json file
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem
            if sess_name in chime6_map["eval"]:
                annotation, scoring_annotation = normalize_chime6(annotation, scoring_txt_normalization,
                                               is_eval=True)
            else:
                annotation, scoring_annotation = normalize_chime6(annotation, scoring_txt_normalization)

            tsplit = None # find chime7 destination split
            for k in ["train", "dev", "eval"]:
                if sess_name in chime6_map[k]:
                    tsplit = k

            with open(os.path.join(out_dir, "transcriptions", tsplit, sess_name + ".json"), "w") as f:
                json.dump(annotation, f, indent=4)
            # retain original annotation but dump also the scoring one
            with open(os.path.join(out_dir, "transcriptions_scoring", tsplit, sess_name + ".json"), "w") as f:
                json.dump(scoring_annotation, f, indent=4)

            # create symlinks too
            [os.symlink(x, os.path.join(out_dir, "audio", tsplit, Path(x).stem) + ".wav") \
             for x in sess2audio[sess_name]]

def prep_dipco(root_dir, out_dir, scoring_txt_normalization="chime7"):

    scoring_txt_normalization = choose_txt_normalization(scoring_txt_normalization)
    if Path(out_dir).exists():
        raise FileExistsError("{} appears to have been created already, "
                               "exiting. Please delete/move/rename "
                              "it if you want to re-generate CHiME7 Task 1 data".format(out_dir))

    def normalize_dipco(annotation, txt_normalizer, is_eval=False):
        annotation_scoring = []
        _get_time = lambda x: (
                dt.strptime(x, "%H:%M:%S.%f") - dt(1900, 1, 1)
        ).total_seconds()
        for indx in range(len(annotation)):
            ex = annotation[indx]
            ex["session_id"] = "S{:02d}".format(dipco_offset + int(ex["session_id"].strip("S")))
            ex["start_time"] = str(_get_time(ex["start_time"]["U01"]))
            ex["end_time"] = str(_get_time(ex["end_time"]["U01"]))
            ex["speaker"] = "P{:02d}".format(dipco_offset + int(ex["speaker_id"].strip("P")))
            del ex["speaker_id"]
            if is_eval:
                new_ex = {}
                for k in ex.keys():
                    if k in ["speaker", "start_time", "end_time", "words", "session_id"]:
                        new_ex[k] = ex[k]
                annotation[indx] = new_ex
                ex = annotation[indx]
                # cannot be used in inference
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    for split in ["dev", "eval"]:
        # here same splits no need to remap
        Path(os.path.join(out_dir, "audio", split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, "transcriptions", split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, "transcriptions_scoring", split)).mkdir(parents=True, exist_ok=True)
        json_dir = os.path.join(root_dir, "transcriptions", split)
        ann_json = glob.glob(os.path.join(json_dir, "*.json"))
        assert len(ann_json) > 0, "DiPCo JSON annotation was not found in {}, please check if " \
                                  "DiPCo data was downloaded correctly and the DiPCo main dir " \
                                  "path is set correctly".format(json_dir)

        # we also create audio files symlinks here
        audio_files = glob.glob(os.path.join(root_dir, "audio", split, "*.wav"))
        sess2audio = {}
        for x in audio_files:
            session_name = Path(x).stem.split("_")[0]
            if session_name not in sess2audio:
                sess2audio[session_name] = [x]
            else:
                sess2audio[session_name].append(x)

        # for each json file
        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem
            if sess_name in dipco_tt:
                annotation, scoring_annotation = normalize_dipco(annotation, scoring_txt_normalization,
                                               is_eval=True)
            else:
                annotation, scoring_annotation = normalize_dipco(annotation, scoring_txt_normalization)
            new_sess_name = "S{:02d}".format(dipco_offset + int(sess_name.strip("S")))
            with open(os.path.join(out_dir, "transcriptions", split, new_sess_name + ".json"), "w") as f:
                json.dump(annotation, f, indent=4)
            with open(os.path.join(out_dir, "transcriptions_scoring", split, new_sess_name + ".json"), "w") as f:
                    json.dump(scoring_annotation, f, indent=4)
            # create symlinks too but swap names for the sessions too
            for x in sess2audio[sess_name]:
                filename = new_sess_name + "_" + "_".join(Path(x).stem.split("_")[1:])
                if filename.split("_")[1].startswith("P"):
                    speaker_id = filename.split("_")[1]
                    filename = filename.split("_")[0] + "_P{:02d}".format(dipco_offset + int(speaker_id.strip("P")))
                os.symlink(x, os.path.join(out_dir, "audio", split, filename + ".wav"))


def prep_mixer6(root_dir, out_dir, eval_gt=False, eval_only=False,
                scoring_txt_normalization="chime7"):

    scoring_txt_normalization = choose_txt_normalization(scoring_txt_normalization)
    if not eval_only and Path(out_dir).exists():
        raise FileExistsError("{} appears to have been created already, "
                               "exiting. Please delete/move/rename "
                              "it if you want to re-generate CHiME7 Task 1 data".format(out_dir))

    def normalize_mixer6(annotation, txt_normalizer, is_eval=False):
        annotation_scoring = []
        for indx in range(len(annotation)):
            ex = annotation[indx]
            ex_scoring = deepcopy(ex)
            ex_scoring["words"] = txt_normalizer(ex["words"])
            if len(ex_scoring["words"]) > 0:
                annotation_scoring.append(ex_scoring)
            # if empty remove segment from scoring
        return annotation, annotation_scoring

    if eval_only:
        splits = ["eval"]
    else:
        splits = ["train_calls", "train_intv",
                  "dev", "eval"]
    audio_files = glob.glob(os.path.join(root_dir,
                                         "export/common/data/corpora/LDC/LDC2013S03/data/pcm_flac",
                                         "**/*.flac"),
                            recursive=True)
    sess2audio = {}
    for x in audio_files:
        session_name = "_".join(Path(x).stem.split("_")[0:-1])
        if session_name not in sess2audio:
            sess2audio[session_name] = [x]
        else:
            sess2audio[session_name].append(x)
    for c_split in splits:
        Path(os.path.join(out_dir, "audio", c_split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, "transcriptions", c_split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, "transcriptions_scoring", c_split)).mkdir(parents=True, exist_ok=True)
        if c_split.startswith("train"):
            ann_json = glob.glob(os.path.join(root_dir, "splits", c_split, "*.json"))
        elif c_split == "dev":
            use_version = "_a" # alternative version is _b see data section
            ann_json = glob.glob(os.path.join(root_dir, "splits", "dev" + use_version, "*.json"))
        elif c_split == "eval":
            if eval_gt == True:
                ann_json = glob.glob(os.path.join(root_dir, "splits", "test", "*.json"))
            else:
                with open(os.path.join(root_dir, "splits", "test.list"), "r") as f:
                    test_list = f.readlines()
                sessions = [x.split("\t")[0] for x in test_list]
                for sess_name in sessions:
                    [os.symlink(x, os.path.join(out_dir, "audio", c_split, Path(x).stem + ".flac")) \
                      for x in sess2audio[sess_name]]
                return

        for j_file in ann_json:
            with open(j_file, "r") as f:
                annotation = json.load(f)
            sess_name = Path(j_file).stem
            if c_split == "eval":
                annotation, annotation_scoring = normalize_mixer6(annotation, scoring_txt_normalization,
                                                 is_eval=True)
            else:
                annotation, annotation_scoring = normalize_mixer6(annotation, scoring_txt_normalization)
            with open(os.path.join(out_dir, "transcriptions", c_split, sess_name + ".json"), "w") as f:
                json.dump(annotation, f, indent=4)
            with open(os.path.join(out_dir, "transcriptions_scoring", c_split, sess_name + ".json"), "w") as f:
                json.dump(annotation_scoring, f, indent=4)
            # create symlinks too
            [os.symlink(x, os.path.join(out_dir, "audio", c_split, Path(x).stem + ".flac")) \
             for x in sess2audio[sess_name]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Data generation script for CHiME-7 Task 1 data.",
                                     add_help=True, usage='%(prog)s [options]')
    parser.add_argument("-c,--chime6_root", type=str, metavar='STR', dest="chime6_root",
                        help="Path to CHiME-6 dataset main directory."
                                                                          "It should contain audio and transcriptions as sub-folders.")
    parser.add_argument("-d,--dipco_root", type=str, metavar='STR', dest="dipco_root",
                        help="Path to DiPCo dataset main directory. "
                                                                         "It should contain audio and transcriptions as sub-folders.")
    parser.add_argument("-m,--mixer6_root", type=str, metavar='STR', dest="mixer6_root",
                        help="Path to DiPCo dataset main directory."
                                                                          "It should contain ")
    parser.add_argument("-o,--output_root", type=str, metavar='STR', dest="output_root",
                        help="Path where the new CHiME-7 Task 1 dataset will be saved."
                             "Note that for audio files symbolic links are used.")
    parser.add_argument("--eval_gt", type=int, default=0, metavar='INT',
                        help="Choose between 0 and 1, 1 will create mixer6 eval annotation also for mixer6 speech, "
                             "it will be released later, ignore for now.")
    parser.add_argument("--txt_norm_scoring", type=str, default="chime6", metavar='STR',
                        help="Choose between chime6 and chime7, this select the text normalization applied when creating" \
                             "the scoring annotation.")
    args = parser.parse_args()
    prep_chime6(args.chime6_root, os.path.join(args.output_root, "chime6"), args.txt_norm_scoring)
    prep_dipco(args.dipco_root, os.path.join(args.output_root, "dipco"), args.txt_norm_scoring)
    prep_mixer6(args.mixer6_root, os.path.join(args.output_root, "mixer6"),
                eval_gt=bool(args.eval_gt),
                scoring_txt_normalization=args.txt_norm_scoring)

