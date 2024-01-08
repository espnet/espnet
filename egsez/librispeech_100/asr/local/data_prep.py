import glob
import os


def create_dataset(data_dir):
    dataset = {}
    for chapter in glob.glob(os.path.join(data_dir, "*/*")):
        text_file = glob.glob(os.path.join(chapter, "*.txt"))[0]

        with open(text_file, "r") as f:
            lines = f.readlines()

        ids_text = {
            line.split(" ")[0]: line.split(" ", maxsplit=1)[1].replace("\n", "")
            for line in lines
        }
        audio_files = glob.glob(os.path.join(chapter, "*.wav"))
        for audio_file in audio_files:
            audio_id = os.path.basename(audio_file)[: -len(".wav")]
            dataset[audio_id] = {"speech": audio_file, "text": ids_text[audio_id]}
    return dataset
