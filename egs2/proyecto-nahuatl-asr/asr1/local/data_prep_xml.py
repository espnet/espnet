import xml.etree.ElementTree as ET
import os
import glob
import re
from thefuzz import process
import random
import sys
import pdb
import string
import subprocess

if len(sys.argv) != 3:
    print("Usage: python data_prep_xml.py [audio_data_path] [data_path]")
    sys.exit(1)
audio_data_path = sys.argv[1]
data_path = sys.argv[2]

# def clean_text(text):
#     # Remove text inside square brackets []
#     text = re.sub(r"\[.*?\]", "", text)
    
#     # Remove double asterisks (**)
#     text = text.replace("**", "")
    
#     # Remove any "*"
#     text = text.replace("*", "")

#     # no skipping paragraph
#     text = text.replace("\n", " ")

#     # Remove "..." or more than three dots
#     text = re.sub(r"\.{3,}", "", text)
    
#     return text.strip()

delset = string.punctuation
delset = delset.replace(":", "")
delset = delset.replace("'", "")


def clean_text(text, text_format="underlying_full"):
    text = re.sub(r"\.\.\.|\*|\[.*?\]|\n", "", text.upper())
    delset_specific = delset
    if text_format == "underlying_full":
        remove_clear = "()=-"
        for char in remove_clear:
            delset_specific = delset_specific.replace(char, "")
    return text.translate(str.maketrans("", "", delset_specific))

def extract_first_channel_16khz(input_wav, output_wav1, output_wav2):
    """Extract the first channel and resample to 16kHz, mono, 16-bit PCM."""
    # check channel
    command = ["soxi", "-c", input_wav]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    num_channels = int(result.stdout.strip())
    if num_channels==1: # mono channel
        if not os.path.exists(output_wav1):
            command = ["sox", input_wav, "-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer", output_wav1]
            subprocess.run(command, check=True)
    else: # multiple channelss    
        if not os.path.exists(output_wav1):   
            command = ["sox", input_wav, "-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer", output_wav1, "remix", "1"]
            subprocess.run(command, check=True)
        if not os.path.exists(output_wav2):
            command = ["sox", input_wav, "-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer", output_wav2, "remix", "2"]
            subprocess.run(command, check=True)

def parse_trans_xml(trans_file, audio_file, out_audio_dir):
    tree = ET.parse(trans_file)
    root = tree.getroot()
    
    text_lines = []
    wav_scp_lines = []
    utt2spk_lines = []
    spk2utt_lines = []
    segments_lines = []
    
    wav_file = os.path.basename(audio_file).strip(".wav")
    num_spk = len(root.findall(".//Speaker"))

    # extract audio wav files
    extract_first_channel_16khz(audio_file, f"{out_audio_dir}/{wav_file}_spk1.wav", f"{out_audio_dir}/{wav_file}_spk2.wav")

    if os.path.exists(f"{out_audio_dir}/{wav_file}_spk1.wav"):
        wav_scp_lines.append(f"{wav_file}_spk1 {out_audio_dir}/{wav_file}_spk1.wav")
    if os.path.exists(f"{out_audio_dir}/{wav_file}_spk2.wav"):
        wav_scp_lines.append(f"{wav_file}_spk2 {out_audio_dir}/{wav_file}_spk2.wav")

    speaker_utt=[] # (start,end,spk,text)
    spk2utt_dict={}

    for turn in root.findall(".//Turn"):
        if "speaker" not in turn.keys():
            continue
        else:
            speaker=turn.attrib["speaker"]
        start_time = turn.get("startTime", "0")
        end_time = turn.get("endTime", "0")

        prev_sync_time=start_time
        text=""
        sync_times = [elem.attrib["time"].strip() for elem in turn.findall("Sync")]
        sync_times.append(end_time)

        prev_sync_time = start_time
        
        spk=speaker
        i=0
        for elem in turn:
            if elem.tag == "Sync":
                sync_time = elem.get("time", "0")
                if elem.tail.strip():
                    speaker_utt.append([sync_times[i],sync_times[i+1],spk,clean_text(elem.tail.strip())])
                    i+=1
            if elem.tag == "Who":
                who_nb = elem.get("nb", "1")
                spk = f"spk{who_nb}"
                speaker_utt.append([sync_times[i],sync_times[i+1],spk,clean_text(elem.tail.strip())])
            if elem.tag == "Comment":
                updated_text = speaker_utt[-1][-1]+" "+clean_text(elem.tail.strip())
                speaker_utt[-1][-1]=updated_text

    for start_time,end_time,spk,text in speaker_utt:
        #print(start_time,end_time,spk,text)
        if start_time=="" or end_time=="" or spk=="" or text=="": continue
        utt_id = f"{wav_file}_{spk}_{start_time}_{end_time}"
        wav_id= f"{wav_file}_{spk}"
        
        if wav_id not in spk2utt_dict:
            spk2utt_dict[wav_id]=[]
        spk2utt_dict[wav_id].append(utt_id)

        utt2spk_lines.append(f"{utt_id} {wav_id}")
        text_lines.append(f"{utt_id} {text}")
        segments_lines.append(f"{utt_id} {wav_id} {start_time} {end_time}")  

    for key,val in spk2utt_dict.items():
        all_utts = " ".join(val)
        curr_line=f"{key} {all_utts}"
        spk2utt_lines.append(curr_line)

    return text_lines, wav_scp_lines, utt2spk_lines, spk2utt_lines, segments_lines

def write_file(content,out_path):
    f=open(out_path,"w")
    f.write("\n".join(content))
    f.close()

random.seed(2025)
# Example usage
trans_prefix={
    "Zacatlan": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Zacatlan-Tepetzintla/Transcripciones-finales",
    "Tequila": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Tequila-Zongolica/Transcripciones-finales",
    "Hidalgo": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Hidalgo-Transcripciones/Transcripciones-Finales"
    }

audio_prefix={
    "Zacatlan": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Zacatlan-Tepetzintla/Grabaciones_Por-dia",
    "Tequila": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Tequila-Zongolica/Grabaciones",
    "Hidalgo": "/ocean/projects/cis210027p/shared/corpora/Proyecto-Nahuatl-ASR/Hidalgo-Grabaciones"
    }

all_text_lines = {"train":[],
                "dev":[],
                "test":{"Zacatlan":[],"Tequila":[],"Hidalgo":[]}
                }
all_wav_scp_lines = {"train":[],
                "dev":[],
                "test":{"Zacatlan":[],"Tequila":[],"Hidalgo":[]}
                }
all_utt2spk_lines = {"train":[],
                "dev":[],
                "test":{"Zacatlan":[],"Tequila":[],"Hidalgo":[]}
                }
all_spk2utt_lines = {"train":[],
                "dev":[],
                "test":{"Zacatlan":[],"Tequila":[],"Hidalgo":[]}
                }
all_segments_lines = {"train":[],
                "dev":[],
                "test":{"Zacatlan":[],"Tequila":[],"Hidalgo":[]}
                }

train_ratio=0.7
dev_ratio=0.8
test_ratio=1.0

for lan in trans_prefix:
    all_trans = sorted(glob.glob(os.path.join(trans_prefix[lan],"*.trs")))
    pattern = r"\d{4}-\d{2}-\d{2}"

    trans_partition = {"train": all_trans[:int(train_ratio*len(all_trans))],
                        "dev": all_trans[int(train_ratio*len(all_trans)):int(dev_ratio*len(all_trans))],
                        "test": all_trans[int(dev_ratio*len(all_trans)):]
                    }

    for curr_set in ["train", "dev", "test"]:
    #for curr_set in ["dev"]:
        for trans in trans_partition[curr_set]:
            matches = re.findall(pattern, trans)
            date_folder = matches[0]
            audio_dir = os.path.join(audio_prefix[lan], date_folder+"*")
            choices = glob.glob(audio_dir+"/*.wav")
            wav=process.extract(trans,choices, limit=1)[0][0]

            text_lines, wav_scp_lines, utt2spk_lines, spk2utt_lines, segments_lines=parse_trans_xml(trans, wav, audio_data_path)
            if curr_set in ["train", "dev"]:
                all_text_lines[curr_set].extend(text_lines)
                all_wav_scp_lines[curr_set].extend(wav_scp_lines)
                all_utt2spk_lines[curr_set].extend(utt2spk_lines)
                all_spk2utt_lines[curr_set].extend(spk2utt_lines)
                all_segments_lines[curr_set].extend(segments_lines)
            else:
                all_text_lines[curr_set][lan].extend(text_lines)
                all_wav_scp_lines[curr_set][lan].extend(wav_scp_lines)
                all_utt2spk_lines[curr_set][lan].extend(utt2spk_lines)
                all_spk2utt_lines[curr_set][lan].extend(spk2utt_lines)
                all_segments_lines[curr_set][lan].extend(segments_lines)                

for curr_set in ["train", "dev"]:
    if not os.path.exists(os.path.join(data_path,curr_set)):
        os.makedirs(os.path.join(data_path,curr_set),exist_ok=True)
    write_file(all_text_lines[curr_set], f"{data_path}/{curr_set}/text")
    write_file(all_wav_scp_lines[curr_set], f"{data_path}/{curr_set}/wav.scp")
    write_file(all_utt2spk_lines[curr_set], f"{data_path}/{curr_set}/utt2spk")
    write_file(all_spk2utt_lines[curr_set], f"{data_path}/{curr_set}/spk2utt")
    write_file(all_segments_lines[curr_set], f"{data_path}/{curr_set}/segments")

for lan in all_text_lines["test"]:
    if not os.path.exists(os.path.join(data_path,"test",lan)):
        os.makedirs(os.path.join(data_path,"test",lan),exist_ok=True)
    write_file(all_text_lines["test"][lan], f"{data_path}/test/{lan}/text")
    write_file(all_wav_scp_lines["test"][lan], f"{data_path}/test/{lan}/wav.scp")
    write_file(all_utt2spk_lines["test"][lan], f"{data_path}/test/{lan}/utt2spk")
    write_file(all_spk2utt_lines["test"][lan], f"{data_path}/test/{lan}/spk2utt")
    write_file(all_segments_lines["test"][lan], f"{data_path}/test/{lan}/segments")
    
