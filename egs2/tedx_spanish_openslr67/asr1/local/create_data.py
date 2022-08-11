import glob
import os


def main(cmd=None):
    # get transcriptions
    transcriptions = {}
    transcripts = open(
        os.environ["TEDX_SPANISH"]
        + "/tedx_spanish_corpus/files/TEDx_Spanish.transcription"
    )

    for row in transcripts:
        txt = " ".join(row.split(" ")[:-1])
        uttid = row.split(" ")[-1].strip()
        transcriptions[uttid] = txt

    splits = ["train", "dev", "test"]

    for split in splits:
        spkrs = open("local/split/" + split + ".txt").read().splitlines()

        utt2spk = open("data/" + split + "/utt2spk", "w")
        wavscp = open("data/" + split + "/wav.scp", "w")
        text = open("data/" + split + "/text", "w")

        for spkr in spkrs:
            for f in glob.glob(
                os.environ["TEDX_SPANISH"]
                + "/tedx_spanish_corpus/speech/"
                + spkr
                + "*.wav"
            ):
                id = f.split("/")[-1][:-4]
                full_id = f[:-4]
                utt2spk.write(id + " " + id + "\n")
                wavscp.write(id + " " + f + "\n")
                txt = transcriptions[id]
                text.write(id + " " + txt + "\n")


if __name__ == "__main__":
    main()
