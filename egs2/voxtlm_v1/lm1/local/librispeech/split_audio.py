import glob
import os
import shutil
import sys

import soundfile as sf
import sox
from tqdm import tqdm


def split_sentence(wav_scps, texts, utt2spks, spk2utts, split_audio_path, id_prefix):
    new_utt2spks = []
    new_spk2utts = []
    new_wav_scps = []
    new_texts = []

    filtered_wav_scps = [line for line in wav_scps if not line.startswith(id_prefix)]
    filtered_utt2spks = [line for line in utt2spks if id_prefix not in line]

    new_wav_scps.append(
        f"{id_prefix}_1 {os.path.abspath(os.path.join(split_audio_path, f'{id_prefix}_1.flac'))}\n"
    )
    new_wav_scps.append(
        f"{id_prefix}_2 {os.path.abspath(os.path.join(split_audio_path, f'{id_prefix}_2.flac'))}\n"
    )
    filtered_wav_scps.extend(new_wav_scps)

    new_utt2spks.append(f"{id_prefix}_1 { '-'.join(id_prefix.split('-')[:2]) }\n")
    new_utt2spks.append(f"{id_prefix}_2 { '-'.join(id_prefix.split('-')[:2]) }\n")
    filtered_utt2spks.extend(new_utt2spks)

    for line in spk2utts:
        if id_prefix in line:
            parts = line.strip().split()
            parts.remove(id_prefix)
            parts.append(id_prefix + "_1")
            parts.append(id_prefix + "_2")
            parts[1:] = sorted(parts[1:])
            new_spk2utts.append(" ".join(parts) + "\n")
        else:
            new_spk2utts.append(line)

    for item in texts:
        if item.startswith(id_prefix):
            new_id_1 = f"{id_prefix}_1"
            new_id_2 = f"{id_prefix}_2"

            new_item_1 = item.replace(id_prefix, new_id_1, 1)
            new_item_2 = item.replace(id_prefix, new_id_2, 1)

            new_texts.append(new_item_1)
            new_texts.append(new_item_2)
        else:
            new_texts.append(item)
    return (
        sorted(filtered_wav_scps),
        sorted(filtered_utt2spks),
        sorted(new_spk2utts),
        sorted(new_texts),
    )


def split_audio_files(asr_eval_path, th=20):
    split_path = os.path.abspath(
        os.path.join(
            os.path.dirname(asr_eval_path), os.path.basename(asr_eval_path) + "_split"
        )
    )
    split_audio_path = os.path.abspath(
        os.path.join(os.path.dirname(asr_eval_path), "tmp_split")
    )
    print("split_path: ", split_path)
    print("split_path: ", split_audio_path)

    wav_scps = open(os.path.join(asr_eval_path, "wav.scp")).readlines()
    utt2spks = open(os.path.join(asr_eval_path, "utt2spk")).readlines()
    spk2utts = open(os.path.join(asr_eval_path, "spk2utt")).readlines()
    texts = open(os.path.join(asr_eval_path, "text")).readlines()

    os.makedirs(split_path, exist_ok=True)
    os.makedirs(split_audio_path, exist_ok=True)

    # shutil.copy(os.path.join(asr_eval_path, 'text'), os.path.join(os.path.dirname(asr_eval_path), os.path.basename(asr_eval_path)+'_split', 'text'))
    i = 0
    new_wav_scps = wav_scps
    new_utt2spks = utt2spks
    new_spk2utts = spk2utts
    new_texts = texts
    for wav_scp in tqdm(wav_scps):
        f = wav_scp.strip().split(" ")[1]
        dur = sox.file_info.duration(f)
        y, sr = sf.read(f)

        assert sr == 16000
        y_len = y.shape[0]

        if y_len > (th * sr):
            i += 1

            # split
            st = 0
            end = y_len
            mid = y_len // 2

            # split file name
            basename = os.path.basename(f)
            new_wav_scps, new_utt2spks, new_spk2utts, new_texts = split_sentence(
                new_wav_scps,
                new_texts,
                new_utt2spks,
                new_spk2utts,
                split_audio_path,
                basename[:-5],
            )
            split_1 = os.path.join(split_audio_path, basename[:-5] + "_1.flac")
            split_2 = os.path.join(split_audio_path, basename[:-5] + "_2.flac")

            sf.write(split_1, y[:mid], sr)
            sf.write(split_2, y[mid + 1 :], sr)

    with open(os.path.join(split_path, "wav.scp"), "w") as split_wav_scp_file:
        for wav_scp in new_wav_scps:
            split_wav_scp_file.write(wav_scp)
    with open(os.path.join(split_path, "utt2spk"), "w") as split_utt2spk_file:
        for utt2spk in new_utt2spks:
            split_utt2spk_file.write(utt2spk)
    with open(os.path.join(split_path, "spk2utt"), "w") as split_spk2utt_file:
        for spk2utt in new_spk2utts:
            split_spk2utt_file.write(spk2utt)
    with open(os.path.join(split_path, "text"), "w") as split_text_file:
        for text in new_texts:
            split_text_file.write(text)

    print("Found: ", i)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script_name.py src_path th")
    else:
        eval_path = sys.argv[1]
        th = int(sys.argv[2])
        split_audio_files(eval_path, th)
