"""The tasks pass --speech_volume_normalize to their preprocessor.

hubert, ssl and slu passed the --rir_scp path instead, so with --rir_scp the
preprocessor multiplied the speech by a string; asr, asr_transducer, s2t and
s2t_ctc read the option only when args had rir_scp.
"""

import importlib

import pytest

TASKS = [
    ("hubert", "HubertTask"),
    ("ssl", "SSLTask"),
    ("slu", "SLUTask"),
    ("asr", "ASRTask"),
    ("asr_transducer", "ASRTransducerTask"),
    ("s2t", "S2TTask"),
    ("s2t_ctc", "S2TTask"),
]


@pytest.fixture
def files(tmp_path):
    rir = tmp_path / "rir.scp"
    rir.write_text("r /dev/null\n")
    tokens = tmp_path / "tokens.txt"
    tokens.write_text(
        "\n".join(
            ["<blank>", "<unk>", "<na>", "<nolang>", "<notimestamps>", "<sos>"]
            + ["<eos>", "<sop>", "<sos/eos>", "a"]
            # the timestamp tokens S2T's preprocessor looks up
            + [f"<{i * 0.02:.2f}>" for i in range(1501)]
        )
        + "\n"
    )
    return rir, tokens


def _preprocess_fn(module, name, extra, tokens):
    task = getattr(importlib.import_module(f"espnet2.tasks.{module}"), name)
    args = task.get_parser().parse_args(
        ["--token_type", "char", "--token_list", str(tokens)] + extra
    )
    return task.build_preprocess_fn(args, train=True)


@pytest.mark.parametrize("module, name", TASKS)
def test_speech_volume_normalize_is_the_option(files, module, name):
    rir, tokens = files
    fn = _preprocess_fn(
        module,
        name,
        ["--rir_scp", str(rir), "--speech_volume_normalize", "0.5"],
        tokens,
    )
    assert fn.speech_volume_normalize == 0.5


@pytest.mark.parametrize("module, name", TASKS)
def test_rir_scp_alone_does_not_normalize(files, module, name):
    rir, tokens = files
    fn = _preprocess_fn(module, name, ["--rir_scp", str(rir)], tokens)
    assert fn.speech_volume_normalize is None
