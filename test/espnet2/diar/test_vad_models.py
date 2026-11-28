"""espnet2.diar.vad_models: the one place that names the VAD models."""

import subprocess
import sys


def test_registry_shape():
    from espnet2.diar.vad_models import VAD_MODELS, VAD_RULES

    assert set(VAD_MODELS) == set(VAD_RULES) == {"online", "offline"}
    for threshold, median in VAD_RULES.values():
        assert 0 < threshold < 1 and median >= 1 and median % 2 == 1


def test_diar_inference_and_cli_use_the_registry():
    from espnet2.bin import cli, diar_inference
    from espnet2.diar import vad_models

    assert diar_inference.VAD_MODELS is vad_models.VAD_MODELS
    assert diar_inference.VAD_RULES is vad_models.VAD_RULES
    assert cli.DEFAULT_MODELS["detect"] == vad_models.VAD_MODELS["offline"]


def test_registry_imports_no_torch():
    """`espnet --help` reads the registry, so it must not load torch."""
    code = (
        "import sys; import espnet2.diar.vad_models; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
