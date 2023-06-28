import argparse
import os
import shutil

from espnet2.fileio.score_scp import MIDReader, SingingScoreWriter

UTT_PREFIX = "csd"
DEV_LIST = ["046"]
TEST_LIST = ["047", "048", "049", "050"]


def train_check(song):
    return not test_check(song) and not dev_check(song)


def dev_check(song):
    for dev in DEV_LIST:
        if dev in song:
            return True
    return False


def test_check(song):
    for test in TEST_LIST:
        if test in song:
            return True
    return False


def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def pack_seg_zero(file_id, number, length=4):
    number = str(number)
    return file_id + "_" + "0" * (length - len(number)) + number


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


def create_score(uid, phns, midis, syb_dur, keep):
    # Transfer into 'score' format
    assert len(phns) == len(midis)
    assert len(midis) == len(syb_dur)
    assert len(syb_dur) == len(keep)
    st = 0
    index_phn = 0
    note_list = []
    while index_phn < len(phns):
        midi = midis[index_phn]
        note_info = [st]
        st += syb_dur[index_phn]
        syb = [phns[index_phn]]
        index_phn += 1
        if (
            index_phn < len(phns)
            and syb_dur[index_phn] == syb_dur[index_phn - 1]
            and midis[index_phn] == midis[index_phn - 1]
            and keep[index_phn] == 0
        ):
            syb.append(phns[index_phn])
            index_phn += 1
        syb = "_".join(syb)
        note_info.extend([st, syb, midi, syb])
        note_list.append(note_info)
        # multi notes in one syllable
        while (
            index_phn < len(phns)
            and keep[index_phn] == 1
            and phns[index_phn] == phns[index_phn - 1]
        ):
            note_info = [st]
            st += syb_dur[index_phn]
            note_info.extend([st, "—", midis[index_phn], phns[index_phn]])
            note_list.append(note_info)
            index_phn += 1
    return note_list


def process_text_info(txt, file_id):
    info = open(txt, "r", encoding="utf-8")
    id = 0
    seg_txt = {}
    for line in info.readlines():
        words = line.strip().split()
        if len(words) == 0:
            continue
        seg_txt[pack_seg_zero(file_id, id)] = words
        id += 1
    return seg_txt


def process_info(csv):
    info = open(csv, "r", encoding="utf-8")
    label_info = []
    phns = []
    midi = []
    for line in info.readlines():
        line = line.strip().split(",")
        if line[0] == "start":
            continue
        start, end, syllable = float(line[0]), float(line[1]), line[3].strip()
        syllable = syllable.replace("_", "-")
        label_info.append("{:.3f} {:.3f} {}".format(start, end, syllable))
        phns.append(syllable)
        midi.append(int(line[2]))
    return label_info, midi, phns


def process_song(notes_list, phns):
    assert len(notes_list) == len(phns)
    index_phn = 0
    note_list = []
    while index_phn < len(phns):
        note = notes_list[index_phn]
        lyric, midi, st, et = note.lyric, note.midi, note.st, note.et
        phn = phns[index_phn]
        note_info = [st, et, phn, midi, phn]
        note_list.append(note_info)
        index_phn = index_phn + 1
    return note_list
    

def process_subset(args, set_name, check_func, dump_dir, fs):
    makedir(os.path.join(args.tgt_dir, set_name))
    
    wavscp = open(os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8")
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open( os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8")
    midscp = open(os.path.join(args.tgt_dir, set_name, "mid.scp"), "w", encoding="utf-8")
    segments = open(os.path.join(args.tgt_dir, set_name, "segments"), "w", encoding="utf-8")

     # get mid_scp
    song_set = []
    for csv in os.listdir(os.path.join(args.src_data, "csv")):

        if not os.path.isfile(os.path.join(args.src_data, "csv", csv)):
            continue
        if not check_func(csv):
            continue
        song_name = csv[:-4]
        song_set.append(song_name)
        
        uid = "{}_{}".format(UTT_PREFIX, pack_zero(song_name))
        midscp.write(
            "{} {}\n".format(uid, os.path.join(args.src_data, "mid", "{}.mid".format(song_name)))
        )
    
    midscp.flush() #refresh buffer
        
    reader = MIDReader(
        fname=os.path.join(args.tgt_dir, set_name, "mid.scp"),
        add_rest=False
    )
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.tgt_dir, set_name, "score.scp")
    )
    
    for song_name in song_set:
        
        uid = "{}_{}".format(UTT_PREFIX, pack_zero(song_name))
        tempo, notes_list = reader[uid]
        label_info, midis, phns = process_info(
            os.path.join(args.src_data, "csv", "{}.csv".format(song_name))
        )
        seg_text = process_text_info(
            os.path.join(args.src_data, "txt", "{}.txt".format(song_name)),
            uid
        )
        note_list = process_song(notes_list, phns)
        
        cmd = "sox -t wavpcm {} -c 1 -t wavpcm -b 16 -r {} {}".format(
            os.path.join(args.src_data, "wav", "{}.wav".format(song_name)),
            fs,
            os.path.join(dump_dir, "csd_{}.wav".format(song_name)),
        )
        os.system(cmd)
        
        for key, val in seg_text.items():
            segments.write("{} {}\n".format(
                key, ' '.join(val)
            ))
        
        wavscp.write(
            "{} {}\n".format(uid, os.path.join(dump_dir, "{}.wav".format(song_name)))
        )
        utt2spk.write("{} {}\n".format(uid, UTT_PREFIX))
        label.write("{} {}\n".format(uid, " ".join(label_info)))
        text.write("{} {}\n".format(uid, (" ".join(phns)).replace('-', ' ')))
        score = dict(
            tempo=tempo, item_list=["st", "et", "lyric", "midi", "phns"], note=note_list
        )
        writer["{}".format(uid)] = score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Oniku Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("train", type=str, help="train set")
    parser.add_argument("dev", type=str, help="development set")
    parser.add_argument("test", type=str, help="test set")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument("--fs", type=int, help="frame rate (Hz)")
    parser.add_argument(
        "--wav_dumpdir", type=str, help=" wav dump directory (rebit)", default="wav_dump"
    )
    parser.add_argument("--g2p", type=str, help="g2p", default="None")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()

    if not os.path.exists(args.wav_dumpdir):
        os.makedirs(args.wav_dumpdir)
        
    process_subset(args, args.train, train_check, args.wav_dumpdir, args.fs)
    process_subset(args, args.dev, dev_check, args.wav_dumpdir, args.fs)
    process_subset(args, args.test, test_check, args.wav_dumpdir, args.fs)
    