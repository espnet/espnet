from pathlib import Path

import pytest
import string
import torch

from espnet2.tasks.enh_asr import EnhASRTask
from espnet2.train.category import UttCategory


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def enh_asr_config_file_1spk(tmp_path: Path, token_list):
    # Write default configuration file
    EnhASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--enh_separator_conf",
            str({"num_spk": 1}),
        ]
    )
    return tmp_path / "asr" / "config.yaml"


@pytest.fixture()
def enh_asr_config_file_2spk(tmp_path: Path, token_list):
    # Write default configuration file
    EnhASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--enh_separator_conf",
            str({"num_spk": 2}),
        ]
    )
    return tmp_path / "asr" / "config.yaml"


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize(
    "utt2category",
    [
        [UttCategory.CLEAN_1SPEAKER],
        [UttCategory.REAL_1SPEAKER],
        [UttCategory.SIMU_DATA],
    ],
)
def test_enh_asr_forward_backward_1spk(enh_asr_config_file_1spk, utt2category):

    shapes = (2, 1000)
    mixture = torch.randn(*shapes)
    speech_ref = torch.randn(*shapes)
    ilens = torch.LongTensor([1000, 800])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    utt2category = torch.LongTensor(utt2category)
    # the default config is for one spk
    espnet_model, _ = EnhASRTask.build_model_from_file(enh_asr_config_file_1spk)
    loss, stats, weight = espnet_model.forward(
        mixture,
        ilens,
        text_ref1=y,
        text_ref1_lengths=y_lens,
        speech_ref1=speech_ref,
        speech_ref1_lengths=ilens,
        utt2category=utt2category,
    )
    loss.backward()


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize(
    "utt2category",
    [
        [UttCategory.CLEAN_1SPEAKER],
        [UttCategory.REAL_1SPEAKER],
        [UttCategory.SIMU_DATA],
    ],
)
def test_enh_asr_forward_backward_2spk(enh_asr_config_file_2spk, utt2category):

    shapes = (2, 1000)
    mixture = torch.randn(*shapes)
    speech_ref1 = torch.randn(*shapes)
    speech_ref2 = torch.randn(*shapes)
    ilens = torch.LongTensor([1000, 800])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    utt2category = torch.LongTensor(utt2category)
    espnet_model, _ = EnhASRTask.build_model_from_file(enh_asr_config_file_2spk)

    loss, stats, weight = espnet_model.forward(
        mixture,
        ilens,
        text_ref1=y,
        text_ref1_lengths=y_lens,
        text_ref2=y,
        text_ref2_lengths=y_lens,
        speech_ref1=speech_ref1,
        speech_ref1_lengths=ilens,
        speech_ref2=speech_ref2,
        speech_ref2_lengths=ilens,
    )
    loss.backward()
