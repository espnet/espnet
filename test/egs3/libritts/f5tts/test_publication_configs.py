"""Guards for the recipe's publication and demo configs.

Loaded over ``egs3/TEMPLATE/f5tts/conf`` the way the template's ``run.py``
does; the template owns the bundle layout and the demo wiring, the recipe
adds its token list.
"""

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config

RECIPE = Path(__file__).resolve().parents[4] / "egs3" / "libritts" / "f5tts"
TEMPLATE_PACKAGE = "egs3.TEMPLATE.f5tts"


def _load(monkeypatch, name, config_name):
    """Load a recipe config the way the template's run.py does."""
    monkeypatch.chdir(RECIPE)
    return load_and_merge_config(
        Path("conf") / name,
        config_name=config_name,
        default_package=TEMPLATE_PACKAGE,
        resolve=False,
    )


def _raw(name):
    """Return the config as plain dicts with interpolations left unresolved."""
    return OmegaConf.to_container(OmegaConf.load(RECIPE / "conf" / name), resolve=False)


def test_publication_config_bundles_the_recipe_token_list(monkeypatch):
    """The bundle must carry `data/tokens`, where this recipe writes its list."""
    publication = _load(monkeypatch, "publication.yaml", "publication.yaml")
    assert "${data_dir}/tokens" in OmegaConf.to_container(
        publication.pack_model.include, resolve=False
    )
    training = _raw("training.yaml")
    assert training["create_token_list"]["save_path"] == "${data_dir}/tokens"


def test_demo_config_needs_no_bundled_code(monkeypatch):
    """The packed inference config names only the system's ``Inference``."""
    cfg = _load(monkeypatch, "demo.yaml", "demo.yaml")
    assert cfg.model.trust_user_code is False
    assert cfg.model.get("call_args") is None
    assert cfg.ui.app_script == "src/app.py"
    assert (RECIPE / "src" / "app.py").is_file()
