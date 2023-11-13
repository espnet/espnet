import argparse
import os

import ffmpeg
import nlp2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def video_to_audio(file_path):
    file_dir, filename_ext = os.path.split(file_path)
    output_dir = f"{str(file_dir)}_wav"
    os.makedirs(output_dir, exist_ok=True)
    filename = filename_ext.split(".")[0]
    if nlp2.is_file_exist(f"{file_dir}/{filename}.mp4"):
        in1 = ffmpeg.input(f"{file_dir}/{filename}.mp4")
        a1 = in1.audio
        out = ffmpeg.output(
            a1,
            f"{output_dir}/{filename}.wav",
            acodec="pcm_s16le",
            ac=1,
            ar="16k",
            vn=None,
        )
        out.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src",
        type=str,
        default="/home/itk0123/s2u/mustard",
        help="Source directory",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=32, help="Number of workers"
    )
    parser.add_argument("-f", "--format", type=str, default="mp4")
    args = parser.parse_args()
    config = vars(args)
    source_dir = config["src"]
    convert_list = []
    for i in tqdm(nlp2.get_files_from_dir(source_dir, match=config["format"])):
        try:
            convert_list.append(i)
        except Exception:
            pass
    # print(len(convert_list))
    process_map(video_to_audio, convert_list, max_workers=config["workers"])
