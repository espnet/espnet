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
@pytest.mark.parametrize("end2endtrain", [True, False])
@pytest.mark.parametrize(
    "utt2category",
    [
        [UttCategory.CLEAN_1SPEAKER],
        [UttCategory.REAL_1SPEAKER],
        [UttCategory.SIMU_DATA],
    ],
)
@pytest.mark.parametrize("enh_real_prob", [0.0, 1.0])
@pytest.mark.parametrize("ctc_weight", [0.0, 1.0])
def test_enh_asr_forward_backward_1spk(
    enh_asr_config_file_1spk, utt2category, end2endtrain, enh_real_prob, ctc_weight
):

    shapes = (2, 1000)
    mixture = torch.randn(*shapes)
    speech_ref = torch.randn(*shapes)
    ilens = torch.LongTensor([1000, 800])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    utt2category = torch.LongTensor(utt2category)
    # the default config is for one spk
    espnet_model, _ = EnhASRTask.build_model_from_file(enh_asr_config_file_1spk)
    espnet_model.end2endtrain = end2endtrain
    espnet_model.enh_real_prob = enh_real_prob
    espnet_model.ctc_weight = ctc_weight
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
@pytest.mark.parametrize("cal_enh_loss", [True, False])
def test_enh_asr_forward_backward_2spk(
    enh_asr_config_file_2spk, utt2category, cal_enh_loss
):

    shapes = (2, 1000)
    mixture = torch.randn(*shapes)
    speech_ref1 = torch.randn(*shapes)
    speech_ref2 = torch.randn(*shapes)
    ilens = torch.LongTensor([1000, 800])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    utt2category = torch.LongTensor(utt2category)
    espnet_model, _ = EnhASRTask.build_model_from_file(enh_asr_config_file_2spk)
    espnet_model.cal_enh_loss = cal_enh_loss
    espnet_model.ctc.reduce = False
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


def test_enh_asr_collect_feats(enh_asr_config_file_1spk):

    shapes = (2, 1000)
    mixture = torch.randn(*shapes)
    ilens = torch.LongTensor([1000, 800])
    espnet_model, _ = EnhASRTask.build_model_from_file(enh_asr_config_file_1spk)
    espnet_model.collect_feats(mixture, ilens)
