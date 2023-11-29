#!/usr/bine/env python3
import sys

from espnet_model_zoo.downloader import ModelDownloader

tts_reference = (
    "@inproceedings{hayashi2020espnet,\n"
    "title={{Espnet-TTS}: Unified, reproducible, "
    "and integratable open source end-to-end text-to-speech toolkit},\n"
    "author={Hayashi, Tomoki and Yamamoto, Ryuichi and Inoue, "
    "Katsuki and Yoshimura, Takenori and Watanabe, Shinji and Toda, "
    "Tomoki and Takeda, Kazuya and Zhang, Yu and Tan, Xu},\n"
    "booktitle={Proceedings of IEEE International Conference on Acoustics, "
    "Speech and Signal Processing (ICASSP)},\n"
    "pages={7654--7658},\n"
    "year={2020},\n"
    "organization={IEEE}\n"
    "}"
)


def create_Readme_file(repo_name, model_name):
    # Fill in the blanks in the template Readme eg. add task tags, model name etc.
    d = ModelDownloader()
    corpus_name = d.query("corpus", name=model_name)[0]
    task_name = d.query("task", name=model_name)[0]
    url_name = d.query("url", name=model_name)[0].split("files/")[0]
    user_name = model_name.split("/")[0]
    lang_name = d.query("lang", name=model_name)[0].replace("jp", "ja")
    template_Readme = open("scripts/TEMPLATE_Readme.md")
    new_Readme = open(repo_name + "/README.md", "w")
    lines_arr = [line for line in template_Readme]
    for line in lines_arr:
        if "<add_more_tags>" in line:
            if task_name == "asr":
                line = line.replace("<add_more_tags>", "automatic-speech-recognition")
            elif task_name == "tts":
                line = line.replace("<add_more_tags>", "text-to-speech")
            elif task_name == "enh":
                line = line.replace(
                    "<add_more_tags>", "speech-enhancement\n- audio-to-audio"
                )
        if "<add_lang>" in line:
            if lang_name == "multilingual":
                line = line.replace("<add_lang>", "en\n- zh\n- ja\n- multilingual")
            else:
                line = line.replace("<add_lang>", lang_name)
        line = line.replace("<add_model_name>", model_name)
        line = line.replace("<add_url>", url_name)
        line = line.replace("<add_name>", user_name)
        line = line.replace("<add_corpus>", corpus_name)
        line = line.replace("<add_task_name>", task_name.upper())
        line = line.replace("<add_recipe_task_name>", task_name.lower() + "1")
        if "<add_tts_reference>" in line:
            if task_name == "tts":
                line = line.replace("<add_tts_reference>", tts_reference)
            else:
                line = line.replace("<add_tts_reference>", "")
        new_Readme.write(line)


if __name__ == "__main__":
    repo_name = sys.argv[1]
    model_name = sys.argv[2]
    create_Readme_file(repo_name, model_name)
