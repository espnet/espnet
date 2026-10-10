"""The template's pack_model and pack_demo configs, run on a tiny recipe.

These go through ``run.py``'s ``main`` the way a user does, then load the
result back the way a user of the published model does.
"""

import shutil
from argparse import Namespace
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import yaml
from hydra.utils import get_class
from omegaconf import OmegaConf

from egs3.TEMPLATE.f5tts.run import main
from espnet3.api.inference import Audio, load
from espnet3.systems.f5tts.system import F5TTSSystem
from espnet3.utils.config_utils import load_default_config


def _run(stages, **configs):
    """Run ``stages`` through the template's ``main`` with the given configs."""
    arguments = {
        "stages": stages,
        "training_config": Path("conf/training.yaml"),
        "inference_config": None,
        "metrics_config": None,
        "publication_config": None,
        "demo_config": None,
        "dry_run": False,
        "write_requirements": False,
    }
    arguments.update({name: Path(path) for name, path in configs.items()})
    main(args=Namespace(**arguments), system_cls=F5TTSSystem)


def _pack_model():
    """Run ``pack_model`` with the recipe's inference and publication configs."""
    _run(
        ["pack_model"],
        inference_config="conf/inference.yaml",
        publication_config="conf/publication.yaml",
    )


def _reference():
    """Reference speech as ``gr.Audio`` hands it over: (rate, int16 samples)."""
    noise = np.random.RandomState(0).randn(22050)
    return 44100, (0.1 * noise * 32767).astype(np.int16)


PACKAGE = "egs3.TEMPLATE.f5tts"


def template_path(*parts) -> Path:
    """Return the path of a file shipped in the template package."""
    return Path(str(resources.files(PACKAGE).joinpath(*parts)))


def test_publication_scaffold_packs_what_an_f5tts_bundle_needs() -> None:
    """The template owns the bundle layout; a recipe only adds its token list."""
    config = load_default_config("publication.yaml", PACKAGE)
    OmegaConf.resolve(config)

    assert config.pack_model.out_dir == "./exp/publication/model_pack"
    # The checkpoint and the config it was trained with both come with
    # `${exp_dir}`; the recipe's dataset code is not needed to run the model.
    assert list(config.pack_model.include) == ["src"]
    assert "last.ckpt" not in config.pack_model.exclude
    assert "inference" in config.pack_model.exclude
    # The model card template ships with the template, not with every recipe.
    assert Path(config.pack_model.readme) == template_path("src", "hf_model_readme.md")
    assert config.upload_model.private is False
    assert config.upload_model["update"] is False  # `.update` is a dict method


def test_demo_scaffold_wires_the_inference_contract_fields() -> None:
    """The demo inputs and outputs are the fields ``Inference`` declares."""
    config = load_default_config("demo.yaml", PACKAGE)
    inference_class = get_class("espnet3.systems.f5tts.inference.Inference")

    assert [(spec.key, spec.type) for spec in config.ui.inputs] == [
        (field.name, field.kind) for field in inference_class.inputs
    ]
    assert [(spec.key, spec.type) for spec in config.ui.outputs] == [
        (field.name, field.kind) for field in inference_class.outputs
    ]
    assert config.ui.app_script == "src/app.py"
    # The packed inference config names only the system's Inference, so the
    # bundle's own code is never imported.
    assert config.model.trust_user_code is False
    assert "call_args" not in config.model
    # The demo needs the training stack to rebuild the model and the vocoder.
    assert any("espnet[train,tts]" in r for r in config.pack.requirements)
    # `exp_tag` arrives from the training config at run time, so resolve only
    # the node under test.
    assert Path(config.pack.readme) == template_path("src", "hf_demo_readme.md")
    assert "- f5-tts" in config.pack.readme_context.tags


def test_pack_model_writes_a_self_contained_bundle(recipe_dir, stub_vocoder):
    """The bundle holds the checkpoint, configs and token list, and no recipe paths."""
    _pack_model()

    bundle = recipe_dir / "exp" / "training" / "model_pack"
    meta = yaml.safe_load((bundle / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["yaml_files"]["inference_config"] == "conf/inference.yaml"

    for kept in (
        "exp/training/last.ckpt",
        "exp/training/config.yaml",
        "conf/training.yaml",
        "data/token_list/tokens.txt",
        "src/app.py",
        "README.md",
    ):
        assert (bundle / kept).is_file(), kept
    assert not (bundle / "dataset").exists()
    for excluded in (
        "exp/training/step40000.ckpt",
        "exp/training/train.log",
        "exp/training/stats",
    ):
        assert not (bundle / excluded).exists(), excluded

    # Nothing the bundle loads points back into the recipe directory.
    for name in ("inference.yaml", "training.yaml"):
        assert str(recipe_dir) not in (bundle / "conf" / name).read_text()
    assert 'load("espnet/minicorpus_f5tts_training")' in (
        bundle / "README.md"
    ).read_text(encoding="utf-8")


def test_packed_model_loads_and_synthesizes_after_moving(
    recipe_dir, stub_vocoder, tmp_path, monkeypatch
):
    """A bundle copied elsewhere loads and synthesizes without the recipe."""
    _pack_model()
    moved = tmp_path / "elsewhere" / "model_pack"
    shutil.copytree(recipe_dir / "exp" / "training" / "model_pack", moved)
    # The recipe is gone: only the bundle is left to load from.
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(recipe_dir)

    model = load(moved)

    assert type(model).__name__ == "Inference"
    assert model.sample_rate == 24000
    output = model("a cab", _reference(), "abba")
    assert isinstance(output["wav"], Audio)
    assert output["wav"].rate == 24000
    assert output["wav"].array.ndim == 1 and output["wav"].array.size > 0


def test_packed_model_needs_no_bundled_code(recipe_dir, stub_vocoder):
    """The packed inference config names only the system's Inference."""
    _pack_model()
    bundle = recipe_dir / "exp" / "training" / "model_pack"

    packed = yaml.safe_load((bundle / "conf" / "inference.yaml").read_text())
    assert packed["model"]["_target_"] == "espnet3.systems.f5tts.inference.Inference"
    for key in ("input_key", "output_fn", "output_artifacts"):
        assert key not in packed, key

    # Loads with the default trust_user_code=False, and runs as the contract
    # says: declared inputs in, an Audio out.
    model = load(bundle)
    output = model("a cab", _reference(), "abba")
    assert isinstance(output["wav"], Audio)
    assert output["wav"].array.dtype == np.float32


def test_pack_demo_builds_a_working_demo(recipe_dir, stub_vocoder):
    """The packed demo builds its Gradio app, example row and handler."""
    gradio = pytest.importorskip("gradio")
    import sys

    _pack_model()
    _run(["pack_demo"], demo_config="conf/demo.yaml")

    demo_dir = recipe_dir / "demo"
    assert (demo_dir / "app.py").read_text() == (
        recipe_dir / "src" / "app.py"
    ).read_text()
    requirements = (demo_dir / "requirements.txt").read_text()
    assert "espnet[train,tts]" in requirements
    demo_config = yaml.safe_load((demo_dir / "demo.yaml").read_text())
    assert demo_config["ui"]["app_script"] == "src/app.py"
    assert demo_config["upload_demo"]["hf_repo"] == "espnet/minicorpus_f5tts_training"
    # The template's input guide and the recipe's example prompt travel along.
    assert (demo_dir / demo_config["ui"]["description"]).is_file()
    assert (demo_dir / "examples" / "prompt.wav").is_file()

    # Import the packed app.py the way a Space runs it: from its own directory.
    sys.path.insert(0, str(demo_dir))
    sys.modules.pop("app", None)
    try:
        import app as packed_app

        blocks = packed_app.build_demo(demo_dir)
    finally:
        sys.modules.pop("app", None)

    # The guide is rendered above the inputs; the example row below them.
    kinds = [type(block).__name__ for block in blocks.blocks.values()]
    assert kinds.index("Markdown") < kinds.index("Textbox") < kinds.index("Dataset")
    (dataset,) = [b for b in blocks.blocks.values() if isinstance(b, gradio.Dataset)]
    assert [row[0] for row in dataset.samples] == ["a cab"]
    # The page opens with the first example filled in.
    (on_load,) = [
        block_function.fn
        for block_function in blocks.fns.values()
        if any(event == "load" for _, event in block_function.targets)
    ]
    text, prompt, transcript = on_load()
    assert (text, transcript) == ("a cab", "abba")
    assert Path(prompt) == (demo_dir / "examples" / "prompt.wav").resolve()

    # gr.Examples registers a fill-in function of its own; the handler is ours.
    (handler,) = [
        block_function.fn
        for block_function in blocks.fns.values()
        if getattr(block_function.fn, "__name__", "") == "synthesize"
    ]
    sample_rate, samples = handler("a cab", _reference(), "abba")
    assert sample_rate == 24000
    assert samples.dtype == np.float32 and samples.ndim == 1 and samples.size > 0
    # The pair is what the output component plays.
    (audio_output,) = [
        block
        for block in blocks.blocks.values()
        if isinstance(block, gradio.Audio) and block.label == "Synthesized speech"
    ]
    assert audio_output.postprocess((sample_rate, samples)) is not None

    with pytest.raises(gradio.Error, match="reference_speech"):
        handler("a cab", None, "abba")
    # An empty transcript box means "not given", which is refused by name.
    with pytest.raises(gradio.Error, match="reference_text"):
        handler("a cab", _reference(), "")
