"""Check the published model card examples against each inference contract."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.publication import InferenceModel
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.publication_utils import _render_readme


@pytest.mark.parametrize(
    "system",
    sorted(
        path.parent.parent.name
        for path in Path("egs3/TEMPLATE").glob("*/conf/publication.yaml")
    ),
)
def test_model_card_usage_matches_inference_contract(system, monkeypatch):
    """A rendered Usage example loads a model and supplies its contract inputs."""
    root = Path("egs3/TEMPLATE") / system
    template = (root / "src/hf_model_readme.md").read_text()
    contract = OmegaConf.load(root / "conf/inference.yaml").contract
    expected_inputs = set(contract.inputs)
    expected_outputs = set(contract.outputs)
    seen = {}

    class FakeModel:
        def __call__(self, sample):
            seen["inputs"] = set(sample) if isinstance(sample, dict) else {"speech"}
            return {
                name: SimpleNamespace(array=None, rate=16000)
                for name in expected_outputs
            }

    def load_model(repo):
        seen["repo"] = repo
        return FakeModel()

    monkeypatch.setattr(InferenceModel, "from_pretrained", load_model)
    rendered = _render_readme(template, {"hf_repo": "example/model"})
    match = re.search(r"## Usage\b.*?```python\n(.*?)\n```", rendered, re.DOTALL)
    assert match is not None, f"{system} has no Python Usage example"
    exec(compile(match.group(1), str(root), "exec"), {})
    assert seen == {"repo": "example/model", "inputs": expected_inputs}


@pytest.mark.parametrize(
    "system",
    sorted(
        path.parent.parent.name
        for path in Path("egs3/TEMPLATE").glob("*/conf/demo.yaml")
    ),
)
def test_demo_card_has_local_launch_commands(system):
    """Each demo card shows how to launch the packed Space locally."""
    template = (Path("egs3/TEMPLATE") / system / "src/hf_demo_readme.md").read_text()
    assert "python -m pip install gradio -r requirements.txt" in template
    assert "python app.py" in template


def test_unknown_output_key_is_detected(monkeypatch):
    """A model card cannot access an output absent from the contract."""
    root = Path("egs3/TEMPLATE/esp2_asr")
    template = (root / "src/hf_model_readme.md").read_text()
    rendered = _render_readme(template, {"hf_repo": "example/model"})
    code = re.search(r"## Usage\s+```python\n(.*?)\n```", rendered, re.DOTALL).group(1)
    monkeypatch.setattr(
        InferenceModel, "from_pretrained", lambda repo: lambda sample: {}
    )
    with pytest.raises(KeyError, match="text"):
        exec(compile(code, str(root), "exec"), {})


def test_vctk_inherits_enhancement_model_card():
    """The VCTK publication overlay uses the enhancement TEMPLATE card."""
    config = load_and_merge_config(
        Path("egs3/vctk_noisy/esp2_enh/conf/publication.yaml"),
        "publication.yaml",
        default_package="egs3.TEMPLATE.esp2_enh",
        resolve=False,
    )
    readme = Path(config.pack_model.readme)
    assert readme.is_file()
    assert readme.as_posix().endswith("egs3/TEMPLATE/esp2_enh/src/hf_model_readme.md")


def test_vctk_inherits_enhancement_demo_card():
    """The VCTK demo overlay uses the enhancement TEMPLATE card."""
    config = load_and_merge_config(
        Path("egs3/vctk_noisy/esp2_enh/conf/demo.yaml"),
        "demo.yaml",
        default_package="egs3.TEMPLATE.esp2_enh",
        resolve=False,
    )
    readme = Path(config.pack.readme)
    assert readme.is_file()
    assert readme.as_posix().endswith("egs3/TEMPLATE/esp2_enh/src/hf_demo_readme.md")


@pytest.mark.parametrize(
    ("recipe", "system"),
    [
        ("librispeech_100", "esp2_asr"),
        ("libritts", "esp2_codec"),
        ("libritts", "f5tts"),
        ("meld", "esp2_cls"),
        ("mini_an4", "esp2_asr"),
        ("must_c", "esp2_st"),
        ("spgispeech", "esp2_asr"),
        ("vctk_noisy", "esp2_enh"),
        ("voxlingua107", "esp2_lid"),
    ],
)
def test_recipe_publication_uses_existing_model_card(recipe, system):
    """Each recipe resolves its publication card from its system TEMPLATE."""
    config = load_and_merge_config(
        Path("egs3") / recipe / system / "conf/publication.yaml",
        "publication.yaml",
        default_package=f"egs3.TEMPLATE.{system}",
        resolve=False,
    )
    assert Path(config.pack_model.readme).is_file()
