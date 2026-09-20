"""Exercise LM input preparation and native command construction locally."""

import gzip
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.systems.asr import language_model as module


def test_native_lm_pipeline_with_external_text(tmp_path, monkeypatch):
    """Keep source text IDs and vocabulary dimensions through all three stages."""
    (tmp_path / "train.txt").write_text("asr_1 HELLO\nasr_empty\n")
    (tmp_path / "valid.txt").write_text("valid_1 WORLD\n")
    (tmp_path / "native.yaml").write_text("lm: seq_rnn\n")
    (tmp_path / "tokens.txt").write_text("<blank>\n<unk>\nHELLO\n<sos/eos>\n")
    (tmp_path / "unigram.model").touch()
    archive = tmp_path / "external.gz"
    with gzip.open(archive, "wt") as stream:
        stream.write("EXTERNAL WORDS\n\nTHIRD LINE\n")
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "lm"),
            "train_text": str(tmp_path / "train.txt"),
            "valid_text": str(tmp_path / "valid.txt"),
            "test_text": str(tmp_path / "valid.txt"),
            "native_config": str(tmp_path / "native.yaml"),
            "tokenizer_dir": str(tmp_path),
            "ngpu": 1,
            "external_text": {"archive": str(archive), "url": "unused"},
        }
    )
    calls = []

    def run(arguments, check):
        calls.append(arguments)
        output = Path(arguments[arguments.index("--output_dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if "--collect_stats" in arguments:
            for split in ("train", "valid"):
                (output / split).mkdir()
                (output / split / "text_shape").write_text("u 2\n")
        elif "espnet2.bin.lm_train" in arguments:
            (output / "valid.loss.ave.pth").touch()
        return None

    monkeypatch.setattr(module.subprocess, "run", run)
    checkpoint = module.train_language_model(config)
    assert checkpoint.is_file()
    assert (checkpoint.parent / "lm_train.txt").read_text().splitlines() == [
        "asr_1 HELLO",
        "librispeech_lng_00000001 EXTERNAL WORDS",
        "librispeech_lng_00000003 THIRD LINE",
    ]
    assert (checkpoint.parent / "stats/train/text_shape.bpe").read_text() == "u 2,4\n"
    assert len(calls) == 3
    assert "espnet2.bin.lm_calc_perplexity" in calls[-1]
