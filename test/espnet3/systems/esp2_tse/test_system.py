"""TSE launcher and stage contract tests."""

import pytest

from espnet3.contract.stages import StageContractError, stage_names
from espnet3.systems.esp2_tse.system import ESP2TSESystem
from espnet3.systems.launch import build_parser, default_conf_package, launch


def test_system_defaults_and_stages():
    assert default_conf_package(ESP2TSESystem) == "egs3.TEMPLATE.esp2_tse"
    assert stage_names(ESP2TSESystem) == [
        "create_dataset",
        "collect_stats",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
        "pack_demo",
        "upload_demo",
    ]
    assert build_parser(ESP2TSESystem).parse_args(["--stages", "infer"]).stages == [
        "infer"
    ]


def test_launcher_requires_stage_config():
    with pytest.raises(StageContractError, match="training"):
        launch(ESP2TSESystem, argv=["--stages", "train"])
