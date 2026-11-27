"""Tests for ESPnet3 SLU task wiring.

``espnet3/systems/esp2_slu/task.py`` is a copy of ``espnet2/tasks/slu.py``
that ``egs3/TEMPLATE/esp2_slu/conf/training.yaml`` names as the task, the way
the other ``esp2_*`` systems name theirs. The first test here keeps the copy
honest; the rest mirror ``test/espnet2/tasks/test_slu.py`` so the copy is
exercised rather than merely imported.
"""

import ast
from argparse import Namespace
from pathlib import Path

import pytest

import espnet2.tasks.slu as upstream_module
import espnet3.systems.esp2_slu.task as copy_module
from espnet3.systems.esp2_slu.task import SLUTask

_DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def _without_docstrings(path: Path) -> str:
    """Return the module's AST dump with every docstring removed.

    The copy differs from espnet2's original only by the docstrings that
    ``ci/test_flake8.sh espnet3`` requires, so comparing the two with those
    stripped is an equality check on everything that runs.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)
    return ast.dump(tree)


def test_copy_still_matches_espnet2():
    """The copy must not drift from the task it copies.

    It has drifted before: master changed `add_task_arguments` to register the
    augmentation options through `add_augmentation_arguments`, and fixed
    `build_preprocess_fn` to pass `speech_volume_normalize` rather than
    `rir_scp`. Neither reached the copy until someone looked.
    """
    assert _without_docstrings(Path(copy_module.__file__)) == _without_docstrings(
        Path(upstream_module.__file__)
    ), (
        "espnet3/systems/esp2_slu/task.py has drifted from espnet2/tasks/slu.py; "
        "copy the original over it and re-add the docstrings."
    )


def test_add_arguments():
    """Ensure parser construction succeeds."""
    SLUTask.get_parser()


def test_add_arguments_help():
    """Ensure parser help exits cleanly."""
    parser = SLUTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    """Ensure main help exits cleanly."""
    with pytest.raises(SystemExit):
        SLUTask.main(cmd=["--help"])


def test_main_print_config():
    """Ensure main print_config exits cleanly."""
    with pytest.raises(SystemExit):
        SLUTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    """Ensure main without args exits with usage."""
    with pytest.raises(SystemExit):
        SLUTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    """Ensure printed config can be parsed back."""
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        SLUTask.print_config(f)
    parser = SLUTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


def test_augmentation_arguments_are_registered():
    """The augmentation options the preprocessor reads must exist.

    They moved into `add_augmentation_arguments` upstream; a copy that missed
    that refactor would silently lose them.
    """
    args = SLUTask.get_parser().parse_args([])
    for name in (
        "rir_scp",
        "rir_apply_prob",
        "noise_scp",
        "noise_apply_prob",
        "noise_db_range",
        "short_noise_thres",
        "speech_volume_normalize",
    ):
        assert hasattr(args, name), name


def _dummy_namespace(**overrides):
    """A minimal SLU namespace; `overrides` replaces single fields."""
    args = Namespace(
        token_type="char",
        token_list=["<blank>", "a", "b", "c", "<space>", "<unk>"],
        transcript_token_list=["<blank>", "a", "b", "c", "<space>", "<unk>"],
        two_pass=False,
        pre_postencoder_norm=False,
        input_size=40,
        frontend="frontend",
        frontend_conf={"n_fft": 51, "win_length": 40, "hop_length": 16},
        specaug="specaug",
        specaug_conf={"apply_time_warp": True, "time_mask_width_range": 4},
        normalize=None,
        normalize_conf=None,
        encoder="transformer",
        encoder_conf={"output_size": 40, "linear_units": 4, "num_blocks": 2},
        decoder="transformer",
        decoder_conf={"linear_units": 4, "num_blocks": 2},
        init=None,
        ctc_conf={},
        model_conf={},
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_build_model():
    """The one-pass model builds with neither post-decoder nor deliberation."""
    model = SLUTask.build_model(_dummy_namespace())
    assert model.postdecoder is None
    assert model.deliberationencoder is None


def test_build_model_two_pass():
    """The two-pass branch wires the post-decoder and deliberation encoder.

    This is the shape `egs3/slurp/esp2_slu` actually trains: a frozen BERT
    post-decoder reads the transcript and a deliberation encoder fuses it with
    the speech encoder output. A tiny stand-in keeps the test cheap; the same
    model is used elsewhere in `test/espnet2`.
    """
    args = _dummy_namespace(
        two_pass=True,
        deliberationencoder="conformer",
        deliberationencoder_conf={
            "output_size": 40,
            "linear_units": 4,
            "num_blocks": 1,
            # `linear`, as the recipe sets: the deliberation encoder takes the
            # speech encoder's output, not raw features, so there is no
            # subsampling front end here.
            "input_layer": "linear",
        },
        postdecoder="hugging_face_transformers",
        postdecoder_conf={
            "model_name_or_path": "hf-internal-testing/tiny-random-BertModel",
            "output_size": 40,
        },
    )
    model = SLUTask.build_model(args)
    assert model.postdecoder is not None
    assert model.deliberationencoder is not None


def test_build_model_rejects_invalid_token_list_type():
    """The vocabulary must be a path or a list."""
    with pytest.raises(RuntimeError, match="token_list must be str or list"):
        SLUTask.build_model(_dummy_namespace(token_list=-1))


def test_build_collate_fn():
    """Ensure the collate function is constructed."""
    SLUTask.build_collate_fn(_dummy_namespace(), True)


@pytest.mark.parametrize("use_preprocessor", [True, False])
def test_build_preprocess_fn(use_preprocessor):
    """Ensure the preprocessor is built, or skipped, as asked."""
    args = _dummy_namespace()
    args.__dict__.update(
        {
            "use_preprocessor": use_preprocessor,
            "bpemodel": None,
            "non_linguistic_symbols": args.token_list,
            "cleaner": None,
            "g2p": None,
        }
    )
    retval = SLUTask.build_preprocess_fn(args, True)
    assert (retval is not None) is use_preprocessor


def test_preprocessor_reads_speech_volume_normalize():
    """`speech_volume_normalize` must reach the preprocessor, not `rir_scp`.

    Upstream passed `rir_scp` here, so setting `--speech_volume_normalize` did
    nothing and setting `--rir_scp` rescaled the waveform instead.
    """
    args = _dummy_namespace()
    args.__dict__.update(
        {
            "use_preprocessor": True,
            "bpemodel": None,
            "non_linguistic_symbols": None,
            "cleaner": None,
            "g2p": None,
            "speech_volume_normalize": 1.0,
            "rir_scp": None,
        }
    )
    assert SLUTask.build_preprocess_fn(args, True).speech_volume_normalize == 1.0


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    """Speech is always required; the target text only outside inference."""
    retval = SLUTask.required_data_names(True, inference)
    assert retval == ("speech",) if inference else ("speech", "text")


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    """The transcript the post-decoder reads is optional either way."""
    assert SLUTask.optional_data_names(True, inference) == ("transcript",)
