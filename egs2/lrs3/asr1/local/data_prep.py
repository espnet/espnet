#!/usr/bin/env python

# Copyright 2022  Debayan Ghosh
#           2022  Carnegie Mellon University
# Apache 2.0


import argparse
import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np


class Utils:
    @staticmethod
    def read_text(text_file: str) -> str:
        """Extracts the transcript from the database-reference text file

        Args:
        text_file (str) : Path to the database-reference text file

        Return:
        (str) The text transcript
        """
        with open(text_file, encoding="ISO-8859-1") as f:
            first_line = f.readline()
        text_val = first_line.split("Text:")[1]
        text_val = text_val.strip("\n")
        text_val = text_val.replace(
            "{LG}", ""
        )  # Special code to avoid scoring seg-fault due to utterance n706Sqp20Mk_50005
        return text_val

    @staticmethod
    def save_list_to_file(list_data: list, save_path: str) -> None:
        """ "Writes content of list_data to a file, line-by-line

        Args:
        list_data: List of Text to be saved to the text file
        save_path: file to save the list_data
        """
        with open(save_path, "w") as f:
            for line in list_data:
                f.write(line + "\n")

    @staticmethod
    def get_parser():
        """Returns the Parser object required to take inputs to data_prep.py"""
        parser = argparse.ArgumentParser(
            description="LRS-3 Data Preparation steps",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--train_val_path", type=str, help="Path to the Train/ Validation files"
        )
        parser.add_argument("--test_path", type=str, help="Path to the Test files")
        return parser


class DatasetUtils:
    @staticmethod
    def train_val_files(
        train_val_path: str, train_val_ratio: float = 0.92, random_seed: int = 0
    ) -> Union[List[str], List[str]]:
        """Splits the folders in 'train_val_path' into the train set and test set,
           and returns the full Train/Validation files.

        Args:
        train_val_path (str): Path to the Folder with the Train/Val data
        train_val_ratio (float): Ratio of the Train/Test file ratio
        random_seed (int): Seed for the file shufling

        Returns:
        speakers_train (list) : Paths of Speaker Folders for Training Data
        speakers_val (list) : Paths of Speaker Folders for Validation Data
        """
        speaker_folders = os.listdir(train_val_path)

        np.random.seed(random_seed)
        np.random.shuffle(speaker_folders)
        num_speakers = len(speaker_folders)

        num_train = int(train_val_ratio * num_speakers)
        speakers_train = speaker_folders[0:num_train]
        speakers_val = speaker_folders[num_train:]

        speakers_train = [
            os.path.join(train_val_path, folder) for folder in speakers_train
        ]
        speakers_val = [os.path.join(train_val_path, folder) for folder in speakers_val]

        return speakers_train, speakers_val

    @staticmethod
    def test_files(test_path: str) -> List[str]:
        """Returns the full path to the Test files

        Args:
        test_path (str): Path to the Folder with the Test data

        Returns:
        speakers_test (list) : Paths of Speaker Folders for Test Data
        """
        speakers_test = os.listdir(test_path)
        speakers_test = [os.path.join(test_path, folder) for folder in speakers_test]
        return speakers_test

    @staticmethod
    def generate_espnet_data(
        speaker_folders: list, dataset: str
    ) -> Union[List[str], List[str], List[str]]:
        """Generates the utt2spk, text and wav data required by ESPNET

        Args:
        speaker_folders (list): The folders from where to extract data
        dataset (str): The dataset we are working with (train, test, dev)

        Returns:
        utt2spk (list) : Utterence to Speaker data
        text (list) : Utterence to Transcript data
        wav (list) : Utterence to Wav-Path data
        """
        utt2spk = []
        text = []
        wav = []

        for speaker_folder in speaker_folders:
            spk_id = os.path.basename(speaker_folder)

            for wav_file in os.listdir(speaker_folder):
                if not wav_file.endswith(".wav"):
                    continue
                text_file = wav_file.replace("wav", "txt")

                wav_full_path = os.path.join(speaker_folder, wav_file)
                text_full_path = os.path.join(speaker_folder, text_file)

                assert os.path.exists(wav_full_path)
                assert os.path.exists(text_full_path)

                utt_id = spk_id + "_" + Path(wav_full_path).stem

                utt2spk.append(utt_id + " " + spk_id)
                wav.append(utt_id + " " + wav_full_path)
                text.append(utt_id + " " + Utils.read_text(text_full_path))
        return utt2spk, text, wav

    @staticmethod
    def perform_data_prep(speaker_folders: list, dataset: str) -> None:
        """Performs ESPNET related Data-Preparation.
        Generates the utt2spk, text and wav.scp files

        Args:
        speaker_folders (list): The folders from where to extract data
        dataset (str): The dataset we are working with (train, test, dev)
        """
        utt2spk, text, wav = DatasetUtils.generate_espnet_data(speaker_folders, dataset)

        utt2spk_file = os.path.join("data", dataset, "utt2spk")
        text_file = os.path.join("data", dataset, "text")
        wav_file = os.path.join("data", dataset, "wav.scp")

        Utils.save_list_to_file(utt2spk, utt2spk_file)
        Utils.save_list_to_file(text, text_file)
        Utils.save_list_to_file(wav, wav_file)


def main():
    parser = Utils.get_parser()
    args = parser.parse_args()
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    train_folders, dev_folders = DatasetUtils.train_val_files(args.train_val_path)
    test_folders = DatasetUtils.test_files(args.test_path)

    logging.info(f"Performing Data Preparation for TEST")
    DatasetUtils.perform_data_prep(test_folders, "test")

    logging.info(f"Performing Data Preparation for TRAIN")
    DatasetUtils.perform_data_prep(train_folders, "train")

    logging.info(f"Performing Data Preparation for DEV")
    DatasetUtils.perform_data_prep(dev_folders, "dev")


if __name__ == "__main__":
    main()
