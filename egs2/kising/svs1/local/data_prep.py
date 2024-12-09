from __future__ import annotations

import argparse
import importlib
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

try:
    from pydub import AudioSegment
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "pydub needed for audio segmentation" "use pip to install pydub"
    ) from error


try:
    import pretty_midi
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "pretty_midi needed for midi data loading" "use pip to install pretty_midi"
    ) from error

from espnet2.fileio.score_scp import SingingScoreWriter

TEST_SET = ["425", "434", "435"]
PREFIX = "kising"
SEGMENT_ID_TEMPLATE = "{prefix}_{singer}_{song_id}_{segid:03}"


@dataclass
class Note:
    onset_time: float
    stop_time: float
    pitch: int  # MIDI pitch
    lyric: str
    phn: str


@dataclass
class Phoneme:
    start_time: float
    stop_time: float
    symbol: str


@dataclass
class KaldiDataset:
    data_folder: Path
    score_dump_folder: Path
    utt2spk: list[tuple[str, str]] = field(default_factory=list)
    utt2wav: list[tuple[str, Path]] = field(default_factory=list)
    utt2text: list[tuple[str, str]] = field(default_factory=list)
    utt2label: list[tuple[str, str]] = field(default_factory=list)
    score_writer: SingingScoreWriter | None = None

    def __post_init__(self):
        self.data_folder = Path(self.data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.score_dump_folder = Path(self.score_dump_folder)
        self.score_dump_folder.mkdir(parents=True, exist_ok=True)
        if self.score_writer is None:
            self.score_writer = SingingScoreWriter(
                outdir=self.score_dump_folder,
                scpfile=self.data_folder / "score.scp",
            )


def load_pinyin_dict(pinyin_dict_path: Path):
    module = importlib.import_module(pinyin_dict_path.stem)
    return module.PINYIN_DICT


def get_partitions(
    input_midi: str | Path, threshold=0.75, max_padding_seconds=0.09
) -> tuple[list[list[float, float]], int]:
    """
    Get partition points [(start, end)] that detach at rests longer than the
    specified `threshold` (in beats). Each segment is padded by
    `max_padding_seconds` (in seconds) at the start and end.
    """
    midi_data = pretty_midi.PrettyMIDI(str(input_midi))
    times, tempo_changes = midi_data.get_tempo_changes()
    assert (
        len(tempo_changes) == 1
    ), f"Unexpected tempo changes for {input_midi}: {tempo_changes}"
    assert times[0] == 0.0, f"Unexpected start time={times[0]} for {input_midi}"
    bpm = tempo_changes[0]

    second_per_beat = 60 / bpm
    thresh_in_second = threshold * second_per_beat

    if len(midi_data.instruments) != 1:
        print(f"Unexpected for kising. Multi-track found: {input_midi}!")
        exit(1)

    partitions = []  # [[start, end], ...]
    instrument = midi_data.instruments[0]  # i.e., just a track
    notes = instrument.notes
    seg_start = max(notes[0].start - max_padding_seconds, 0.0)
    prev_note_end = notes[0].end

    for note in notes[1:]:
        if note.start - prev_note_end >= thresh_in_second:
            silence_midpoint = (prev_note_end + note.start) / 2
            partitions.append(
                [seg_start, min(silence_midpoint, prev_note_end + max_padding_seconds)]
            )
            seg_start = max(note.start - max_padding_seconds, silence_midpoint)
        prev_note_end = note.end

    # NOTE: The end of the last partition can be larger than the
    # corresponding audio length
    partitions.append([seg_start, notes[-1].end + 0.2])

    return partitions, int(bpm)


def save_wav_segments_from_partitions(
    audio: AudioSegment,
    partitions: list[tuple[float, float]],
    output_directory: str | Path,
    song_id: str,
    singer: str,
) -> tuple[list[str], list[Path], list[float]]:
    """
    Partition WAV files based on 'partitions' and save the segmented files.
    """

    assert audio.channels in {
        1,
        2,
    }, f"Unexpected number of channels in kising-v2: {audio.channels}"

    if audio.channels == 2:
        audio = audio.split_to_mono()[1]  # NOTE: use the right channel when it's stereo

    segids = []
    output_filenames = []
    durations = []

    for i, (start_time, end_time) in enumerate(partitions, 1):
        start_time_ms = int(start_time * 1000)
        end_time_ms = int(end_time * 1000)

        segment = audio[start_time_ms:end_time_ms]
        segid = SEGMENT_ID_TEMPLATE.format(
            prefix=PREFIX, singer=singer, song_id=song_id, segid=i
        )
        outfile = output_directory / f"{segid}.wav"

        segment.export(outfile, format="wav")
        segids.append(segid)
        output_filenames.append(outfile)
        durations.append(len(segment) / 1000.0)

    return segids, output_filenames, durations


# Function to split Pinyin into shengmu and yunmu
def split_pinyin(pinyin: str, pinyin_dict: dict) -> tuple[str]:
    # load pinyin dict from local/pinyin.dict
    pinyin = pinyin.lower()
    if pinyin not in pinyin_dict:
        logging.debug(f"Unknown pinyin: {pinyin} in pinyin_dict")
        return (pinyin,)
    return pinyin_dict[pinyin]


def get_info_from_partitions(
    midi_path: str | Path,
    partitions: list[tuple[float, float]],
    pinyin_dict: dict[str, tuple[str]],
) -> tuple[list[list[Note]], list[list[Phoneme]]]:
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    notes = midi_data.instruments[0].notes
    lyrics = midi_data.lyrics

    assert len(notes) == len(lyrics), "Mismatch between notes and lyrics length"
    assert all(
        [note.start == lyric.time for note, lyric in zip(notes, lyrics)]
    ), f"Mismatch between notes and lyrics start time. Got {notes=}, {lyrics=}"
    assert all([start < end for start, end in partitions]), (
        f"Partition start time should be smaller than end time. Got {partitions=}."
        "This issue could be due to incomplete audio data."
        "Try deleting the current audio folder and rerunning the script."
    )
    assert all(
        [partitions[i][1] <= partitions[i + 1][0] for i in range(len(partitions) - 1)]
    ), f"Partitions should be non-overlapping and increasing in time. Got {partitions=}"

    partition_notes = []
    partition_phns = []

    lyric_index = 0
    for partition_start, partition_end in partitions:
        current_notes = []
        current_phns = []

        # Skip lyrics before partition_start
        while lyrics[lyric_index].time < partition_start:
            lyric_index += 1

        # At the start of the partition
        if (lyrics[lyric_index].time - partition_start) > 0:
            current_notes.append(
                Note(
                    onset_time=0,
                    stop_time=lyrics[lyric_index].time - partition_start,
                    pitch=0,
                    lyric="<AP>",
                    phn="<AP>",
                )
            )
            current_phns.append(
                Phoneme(
                    start_time=0,
                    stop_time=lyrics[lyric_index].time - partition_start,
                    symbol="<AP>",
                )
            )

        # Iterate through lyrics until partition_end
        while lyric_index < len(lyrics) and lyrics[lyric_index].time < partition_end:
            lyric = lyrics[lyric_index]
            note = notes[lyric_index]
            pitch = note.pitch
            note_duration = note.end - note.start
            note_relative_start = note.start - partition_start
            note_relative_end = note.end - partition_start

            if lyric.text in {"?", "-"}:
                # Slur
                current_notes.append(
                    Note(
                        onset_time=note_relative_start,
                        stop_time=note_relative_end,
                        pitch=pitch,
                        lyric=lyric.text,
                        phn=current_phns[-1].symbol,
                    )
                )
                current_phns.append(
                    Phoneme(
                        start_time=note_relative_start,
                        stop_time=note_relative_end,
                        symbol=current_phns[-1].symbol,
                    )
                )
            elif lyric.text.isalpha():
                phns_in_lyric = split_pinyin(lyric.text, pinyin_dict)
                current_notes.append(
                    Note(
                        onset_time=note_relative_start,
                        stop_time=note_relative_end,
                        pitch=pitch,
                        lyric="_".join(phns_in_lyric),
                        phn="_".join(phns_in_lyric),
                    )
                )
                if len(phns_in_lyric) == 1:
                    current_phns.append(
                        Phoneme(
                            start_time=note_relative_start,
                            stop_time=note_relative_end,
                            symbol=phns_in_lyric[0],
                        )
                    )
                elif len(phns_in_lyric) == 2:
                    initial_vowel_duration_split = random.uniform(0.2, 0.25)  # 1:5, 1:4
                    initial_relative_split_end = (
                        note_relative_start
                        + note_duration * initial_vowel_duration_split
                    )
                    current_phns += [
                        Phoneme(
                            start_time=note_relative_start,
                            stop_time=initial_relative_split_end,
                            symbol=phns_in_lyric[0],
                        ),
                        Phoneme(
                            start_time=initial_relative_split_end,
                            stop_time=note_relative_end,
                            symbol=phns_in_lyric[1],
                        ),
                    ]
                else:
                    raise ValueError(
                        f"Unexpected number of phonemes in lyric: {lyric.text}"
                    )

            lyric_index += 1

        # at the end of the partition
        if (partition_end - notes[lyric_index - 1].end) > 0:
            current_notes.append(
                Note(
                    onset_time=notes[lyric_index - 1].end - partition_start,
                    stop_time=partition_end - partition_start,
                    pitch=0,
                    lyric="<SP>",
                    phn="<SP>",
                )
            )
            current_phns.append(
                Phoneme(
                    start_time=notes[lyric_index - 1].end - partition_start,
                    stop_time=partition_end - partition_start,
                    symbol="<SP>",
                )
            )

        partition_notes.append(current_notes)
        partition_phns.append(current_phns)

    return partition_notes, partition_phns


def process_dataset(args):
    trainset = KaldiDataset(args.tgt_dir / f"train_{args.dataset}", args.score_dump)
    testset = KaldiDataset(args.tgt_dir / f"test_{args.dataset}", args.score_dump)

    pinyin_dict = load_pinyin_dict(args.pinyin_dict)

    for song_folder in args.src_data.iterdir():
        if (not song_folder.is_dir()) or song_folder.stem.endswith("-unseg"):
            continue

        try:
            song = int(song_folder.name[:3])
        except ValueError:
            logging.error(f"Invalid song number in folder name: {song_folder}")
            continue
        # Skip songs between 436 and 440
        if song >= 436 and song <= 440:
            continue

        wav_files = list(song_folder.glob("*.wav"))
        if not wav_files:
            logging.error(f"No wav files found in {song_folder}")
            continue
        midi_files = list(song_folder.glob("*.mid"))
        if len(midi_files) != 1:
            logging.error(
                f"Unexpected number of midi files in {song_folder}: {len(midi_files)=}"
            )
            continue

        logging.info(f"Processing: {song_folder}")
        midi_file = midi_files[0]
        song_id = song_folder.stem
        partitions, tempo = get_partitions(midi_file)
        for wav_file in wav_files:
            match = re.search(r"([0-9-]+)-([a-zA-Z-]+)", wav_file.stem)
            assert song_id == match.group(1)
            singer = match.group(2)
            if (args.dataset == "acesinger" and singer == "original") or (
                args.dataset == "original" and singer != "original"
            ):
                continue

            audio = AudioSegment.from_wav(wav_file)
            # truncate the last partition if it's longer than the audio
            partitions[-1][1] = min(partitions[-1][1], audio.duration_seconds)
            segids, filenames, _ = save_wav_segments_from_partitions(
                audio, partitions, args.wav_dumpdir, song_id, singer
            )
            partitioned_notes, partitioned_phns = get_info_from_partitions(
                midi_file, partitions, pinyin_dict
            )
            if song_id in TEST_SET:
                dataset = testset
            else:
                dataset = trainset
            dataset.utt2spk += [(segid, singer) for segid in segids]
            dataset.utt2wav += list(zip(segids, filenames))
            dataset.utt2text += [
                (segid, *[phn.symbol for phn in phns])
                for segid, phns in zip(segids, partitioned_phns)
            ]
            dataset.utt2label += [
                (
                    segid,
                    *[
                        f"{phn.start_time:.3f} {phn.stop_time:.3f} {phn.symbol}"
                        for phn in phns
                    ],
                )
                for segid, phns in zip(segids, partitioned_phns)
            ]
            for segid, notes in zip(segids, partitioned_notes):
                dataset.score_writer[segid] = {
                    "tempo": tempo,
                    "item_list": ["st", "et", "lyric", "midi", "phns"],
                    "note": [
                        [
                            round(note.onset_time, 3),
                            round(note.stop_time, 3),
                            note.lyric,
                            note.pitch,
                            note.phn,
                        ]
                        for note in notes
                    ],
                }

    for dataset in [trainset, testset]:
        with open(dataset.data_folder / "utt2spk", "w") as outfile:
            write_kaldi_file(outfile, dataset.utt2spk)
        with open(dataset.data_folder / "wav.scp", "w") as outfile:
            write_kaldi_file(outfile, dataset.utt2wav)
        with open(dataset.data_folder / "text", "w") as outfile:
            write_kaldi_file(outfile, dataset.utt2text)
        with open(dataset.data_folder / "label", "w") as outfile:
            write_kaldi_file(outfile, dataset.utt2label)


def write_kaldi_file(fhand: TextIO, tuples: list[tuple]):
    fhand.writelines([" ".join(map(str, tup)) + "\n" for tup in sorted(tuples)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for KiSing Database")
    parser.add_argument("src_data", type=Path, help="source data directory")
    parser.add_argument("--tgt_dir", type=Path, default="data")
    parser.add_argument(
        "--wav_dumpdir",
        type=Path,
        help="wav dump directory (rebit)",
        default="wav_dump",
    )
    parser.add_argument(
        "--score_dump", type=Path, default="score_dump", help="score dump directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset to process (original|acesinger|all)",
    )
    parser.add_argument(
        "--pinyin_dict",
        type=Path,
        default=Path("local/pinyin_dict2.py"),
        help="pinyin dict file",
    )
    args = parser.parse_args()

    logging.debug(args)
    logging.info(f"Start processing data from {args.src_data}")

    process_dataset(args)
