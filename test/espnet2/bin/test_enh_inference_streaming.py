from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
import yaml

from espnet2.bin.enh_inference_streaming import (
    SeparateSpeechStreaming,
    get_parser,
    main,
)
from espnet2.tasks.enh import EnhancementTask
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def config_file(tmp_path: Path):
    # Write default configuration file
    EnhancementTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "enh"),
        ]
    )

    with open(tmp_path / "enh" / "config.yaml", "r") as f:
        args = yaml.safe_load(f)

    args.update(
        {
            "encoder": "stft",
            "encoder_conf": {"n_fft": 512, "hop_length": 256},
            "decoder": "stft",
            "decoder_conf": {"n_fft": 512, "hop_length": 256},
            "separator": "skim",
            "separator_conf": {
                "causal": True,
                "seg_overlap": False,
                "num_spk": 2,
                "unit": 128,
                "layer": 2,
            },
        }
    )

    with open(tmp_path / "enh" / "config.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)

    return tmp_path / "enh" / "config.yaml"


@pytest.mark.execution_timeout(10)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("input_size", [8000, 16000])
def test_SeparateSpeech(
    config_file,
    batch_size,
    input_size,
):
    separate_speech = SeparateSpeechStreaming(
        train_config=config_file,
    )
    wav = torch.rand(batch_size, input_size)
    ilens = torch.LongTensor([input_size])

    speech_sim_chunks = separate_speech.frame(wav)
    output_chunks = [[] for ii in range(separate_speech.num_spk)]

    for chunk in speech_sim_chunks:
        output = separate_speech(chunk)
        for channel in range(separate_speech.num_spk):
            output_chunks[channel].append(output[channel])

    separate_speech.reset()
    waves = [separate_speech.merge(chunks, ilens) for chunks in output_chunks]
