"""
Split a MIDI file and its corresponding wav files base on rests.
Save processed files in output_dir
"""

import os
from pydub import AudioSegment
import pretty_midi
import re
from typing import TextIO, List, Tuple


def get_partitions(input_midi: str, threshold=2.) -> List[Tuple[float, float]]:
    """"
    Get partition points [(start, end)] that detach at rests longer than the specified threshold.
    Note: silence truncated when longer than 0.4s.

    Parameters
    -------
    input_midi : str
        Path to the MIDI file.
    threshold : float
        The threshold in beats. Rests longer than this threshold will trigger detachment.

    Returns
    -------
    partitions : List[Tuple[float, float]]
        The partition points. Each entry of the list is a tuple of (start, end)
    """
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    bpm = midi_data.estimate_tempo()
    second_per_beat = 60 / bpm
    thresh_in_second = threshold * second_per_beat

    if len(midi_data.instruments) != 1:
        print(f"Unexpected for kising. Multi-track found: {input_midi}!")
        exit(1)

    partitions = []  # [(start, end)]
    instrument = midi_data.instruments[0]  # i.e., just a track
    notes = instrument.notes
    seg_start = max(notes[0].start - 0.2, 0.)
    prev_note_end = notes[0].end

    for note in notes[1:]:
        if note.start - prev_note_end >= thresh_in_second:
            partitions.append(
                (seg_start, min((prev_note_end + note.start) / 2, prev_note_end + 0.2)))
            seg_start = max(note.start - 0.2, (prev_note_end + note.start) / 2)
        prev_note_end = note.end

    # Note: the end of the last partition can be larger than the corresponding audio length. But does not matter
    partitions.append((seg_start, notes[-1].end + 0.2))

    return partitions


def save_wav_segments_from_partitions(input_wav, partitions, outdir, songid, singer):
    """
    Partition wav files based on 'partitions' and save the segmented files in 'outdir'.

    Returns
    --------
    tuple
        A tuple containing two lists:
        - spk2utt (for singer)
        - wav_scp
    """
    audio = AudioSegment.from_wav(input_wav)
    segids = []
    wav_scp = []

    for i, (start_time, end_time) in enumerate(partitions, 1):
        start_time_ms = int(start_time * 1000)
        end_time_ms = int(end_time * 1000)

        # Note: end_time_ms can be larger than the actual audio length. But does not matter here
        segment = audio[start_time_ms:end_time_ms]
        outfile = os.path.join(outdir, f"{songid}_{i:03}_{singer}.wav")
        segid = f"{songid}_{i:03}_{singer}"

        segment.export(outfile, format="wav")
        segids.append(segid)
        wav_scp.append((segid, outfile))

    return segids, wav_scp


def get_text_from_partitions(input_midi: str, partitions: List[Tuple[float, float]]) -> List[List[str]]:
    """"
    Parameters
    --------
    input_midi
    partitions: non-overlapping time partitions with increasing timestamps
    """
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    current_partition = 0
    text = [[]]
    for lyric in midi_data.lyrics:
        assert lyric.time > partitions[current_partition][0]
        if lyric.time > partitions[current_partition][1]:
            text.append([])
            current_partition += 1
            assert lyric.time > partitions[current_partition][0]
            assert lyric.time < partitions[current_partition][1]
        if lyric.text != '-':
            if '#' in lyric.text:
                # Note: treat "word#1", "word#2",... as one single word. I.e, treat one word sung with distinct notes as 1 word
                assert len(lyric.text.split('#')) == 2
                word, id = lyric.text.split('#')
                if id == '1':
                    text[-1].append(word)
            else:
                text[-1].append(lyric.text)
    return text


def append_spk2utt(fhand: TextIO, singer, segids):
    fhand.write(' '.join([singer] + segids) + '\n')


def append_utt2spk(fhand: TextIO, singer, segids):
    fhand.writelines([' '.join((segid, singer)) + '\n' for segid in segids])


def append_wav_scp(fhand: TextIO, wav_scp):
    fhand.writelines([' '.join(segid2path) + '\n' for segid2path in wav_scp])


def append_text(fhand: TextIO, segids, text):
    fhand.writelines(' '.join([segid] + t) +
                     '\n' for segid, t in zip(segids, text))


if __name__ == "__main__":
    import glob
    import shutil
    # import sys

    # if len(sys.argv) != 3:
    #     print("Usage: python3 data_prep.py dataset_dir output_dir")
    #     sys.exit(1)

    # dataset_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    dataset_dir = "/NASdata/AudioData/KiSing-v2"
    output_dir = "downloads"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    wav_dir = os.path.join(output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=False)

    spk2utt_filepath = os.path.join(output_dir, "spk2utt")
    utt2spk_filepath = os.path.join(output_dir, "utt2spk")
    wav_scp_filepath = os.path.join(output_dir, "wav.scp")
    text_filepath = os.path.join(output_dir, "text")
    spk2utt_file = open(spk2utt_filepath, "w")
    utt2spk_file = open(utt2spk_filepath, "w")
    wav_scp_file = open(wav_scp_filepath, "w")
    text_file = open(text_filepath, "w")

    print("Data preparation starts")
    for subdir in os.listdir(dataset_dir):
        full_subdir_path = os.path.join(dataset_dir, subdir)

        if os.path.isdir(full_subdir_path) and "-unseg" not in subdir:
            print(f"Processing: {full_subdir_path}")
            wav_files = glob.glob(os.path.join(full_subdir_path, '*.wav'))
            midi_file = glob.glob(os.path.join(full_subdir_path, '*.mid'))[0]
            partitions = get_partitions(midi_file)
            songid = subdir

            for wav_file in wav_files:
                match = re.search(r'([0-9-]+)-([a-zA-Z-]+)\.wav', wav_file)
                assert songid == match.group(1)
                singer = match.group(2)
                wav_singer_dir = os.path.join(wav_dir, singer)
                os.makedirs(wav_singer_dir, exist_ok=True)
                segids, wav_scp = save_wav_segments_from_partitions(
                    wav_file, partitions, wav_singer_dir, songid, singer)
                text = get_text_from_partitions(midi_file, partitions)
                append_spk2utt(spk2utt_file, singer, segids)
                append_utt2spk(utt2spk_file, singer, segids)
                append_wav_scp(wav_scp_file, wav_scp)
                append_text(text_file, segids, text)

    spk2utt_file.close()
    utt2spk_file.close()
    wav_scp_file.close()
    text_file.close()
