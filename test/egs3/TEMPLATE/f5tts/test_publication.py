"""The template's pack_model and pack_demo configs, run on a tiny recipe.

These go through ``launch`` the way a recipe's ``run.py`` does, then load the
result back the way a user of the published model does.
"""

import shutil
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from espnet3.contract.kinds.audio import Audio
from espnet3.publication import InferenceModel
from espnet3.systems.f5tts.system import F5TTSSystem
from espnet3.systems.launch import launch
from espnet3.utils.config_utils import load_default_config


def _run(stages, **configs):
    """Run ``stages`` through ``launch`` with the given configs."""
    argv = ["--stages", *stages, "--training_config", "conf/training.yaml"]
    for name, path in configs.items():
        argv += [f"--{name}", str(path)]
    launch(F5TTSSystem, argv=argv)


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


def _item(reference=None):
    """One sample the way ``InferenceModel`` and the demo pass it: by field name."""
    return {
        "text": "a cab",
        "reference_speech": _reference() if reference is None else reference,
        "reference_text": "abba",
    }


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


def test_demo_scaffold_leaves_the_widgets_to_the_inference_contract() -> None:
    """The demo declares no widgets of its own; the session reads the contract."""
    config = load_default_config("demo.yaml", PACKAGE)

    assert config.ui.get("inputs") is None
    assert config.ui.get("outputs") is None
    assert config.ui.app_script == "src/app.py"
    # The packed inference config names only the system's Inference and
    # declares its contract inline, so the bundle's own code is never imported.
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
    assert 'InferenceModel.from_pretrained("espnet/minicorpus_f5tts_training")' in (
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

    model = InferenceModel.from_packed(moved)

    assert type(model.model).__name__ == "Inference"
    assert model.model.sample_rate == 24000
    output = model(_item())
    assert isinstance(output["wav"], Audio)
    assert output["wav"].rate == 24000
    assert output["wav"].array.dtype == np.float32
    assert output["wav"].array.ndim == 1 and output["wav"].array.size > 0


def test_packed_model_needs_no_bundled_code(recipe_dir, stub_vocoder):
    """The packed config names only the system's Inference and its contract.

    Nothing in the bundle itself is imported, so it loads untrusted; the
    contract travels in the config, not in code.
    """
    _pack_model()
    bundle = recipe_dir / "exp" / "training" / "model_pack"

    packed = yaml.safe_load((bundle / "conf" / "inference.yaml").read_text())
    assert (
        packed["model"]["_target_"] == "espnet3.systems.f5tts.src.inference.Inference"
    )
    assert packed["input_key"] == ["text", "reference_speech", "reference_text"]
    assert "output_fn" not in packed
    assert list(packed["contract"]["inputs"]) == packed["input_key"]
    assert packed["contract"]["outputs"]["wav"]["sample_rate"] == 24000
    assert "imports" not in packed["contract"]

    # Loads with the default trust_user_code=False, and runs: the three
    # inputs in by name, the utt_id-stamped record out, the waveform as an
    # Audio at the contract's rate.
    model = InferenceModel.from_packed(bundle)
    assert model.contract is not None
    output = model(_item(), idx="u0")
    assert output["utt_id"] == "u0"
    assert output["wav"].rate == 24000
    assert output["wav"].array.dtype == np.float32
    # A path is read and resampled by the contract like the Gradio pair, and
    # a stereo pair is reduced to its first channel rather than refused.
    import soundfile as sf

    prompt = recipe_dir / "prompt16k.wav"
    sf.write(prompt, np.zeros(16000, dtype=np.float32), 16000)
    assert model(_item(str(prompt)))["wav"].array.ndim == 1
    rate, mono = _reference()
    stereo = (rate, np.stack([mono, mono], axis=1))
    assert model(_item(stereo))["wav"].array.ndim == 1
    # The contract refuses what is not audio, by the input's name.
    with pytest.raises(TypeError, match="audio"):
        model(_item(42))
    with pytest.raises(TypeError, match="unknown"):
        model({**_item(), "unknown": 1})


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
    assert "inputs" not in demo_config["ui"]  # the widgets come from the contract
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

    # The guide is rendered above the inputs; the example row below them. The
    # widgets and their labels are the contract's.
    kinds = [type(block).__name__ for block in blocks.blocks.values()]
    assert kinds.index("Markdown") < kinds.index("Textbox") < kinds.index("Dataset")
    labels = [
        block.label
        for block in blocks.blocks.values()
        if isinstance(block, (gradio.Textbox, gradio.Audio))
    ]
    assert labels == [
        "Text to synthesize",
        "Reference speech (voice to clone)",
        "Reference transcript",
        "Synthesized speech",
    ]
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
    # The session turns the model's Audio into the pair gr.Audio plays.
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

    # An empty widget is reported by the label the page shows, before the
    # model is called; an empty transcript box counts as empty.
    with pytest.raises(gradio.Error, match="Reference speech"):
        handler("a cab", None, "abba")
    with pytest.raises(gradio.Error, match="Reference transcript"):
        handler("a cab", _reference(), "  ")
    # What the contract refuses is shown in the UI too.
    with pytest.raises(gradio.Error, match="audio"):
        handler("a cab", 42, "abba")
