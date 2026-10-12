"""Tests for espnet3.components.contract.stages."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.components.contract.stages import (
    StageContractError,
    StageSpec,
    check_requested_stages,
    check_stage_contract,
    roles,
    stage_log_dir,
    stage_names,
    stage_spec,
)

# ---------------------------------------------------------------------------
# StageSpec: keyword-only construction
# ---------------------------------------------------------------------------


def test_stagespec_rejects_positional_arguments():
    with pytest.raises(TypeError):
        StageSpec("train", "training")


# ---------------------------------------------------------------------------
# check_stage_contract: definition-time errors
# ---------------------------------------------------------------------------


def test_check_stage_contract_rejects_empty_stages():
    class Empty:
        stages = ()

    with pytest.raises(StageContractError, match="must be a non-empty tuple"):
        check_stage_contract(Empty)


def test_check_stage_contract_rejects_non_stagespec_elements():
    class NotASpec:
        stages = ("train",)

        def train(self):
            pass

    with pytest.raises(StageContractError, match="must be a non-empty tuple"):
        check_stage_contract(NotASpec)


def test_check_stage_contract_rejects_duplicate_names():
    class Duplicate:
        stages = (
            StageSpec(name="train", config="training"),
            StageSpec(name="train", config="training"),
        )

        def train(self):
            pass

    with pytest.raises(StageContractError, match="repeats a name"):
        check_stage_contract(Duplicate)


def test_check_stage_contract_accepts_any_identifier_config_role():
    """Roles aren't a fixed list anymore; any non-empty identifier is a role."""

    class CustomRole:
        stages = (StageSpec(name="train", config="not_a_predeclared_role"),)

        def train(self):
            pass

    check_stage_contract(CustomRole)


def test_check_stage_contract_rejects_empty_config_role():
    class EmptyRole:
        stages = (StageSpec(name="train", config=""),)

        def train(self):
            pass

    with pytest.raises(StageContractError, match="non-empty identifier"):
        check_stage_contract(EmptyRole)


def test_check_stage_contract_rejects_non_identifier_config_role():
    class BadRole:
        stages = (StageSpec(name="train", config="not-an-identifier"),)

        def train(self):
            pass

    with pytest.raises(StageContractError, match="non-empty identifier"):
        check_stage_contract(BadRole)


def test_check_stage_contract_rejects_non_str_log_dir():
    class BadLogDir:
        stages = (StageSpec(name="train", config="training", log_dir=123),)

        def train(self):
            pass

    with pytest.raises(StageContractError, match="log_dir must be a str or None"):
        check_stage_contract(BadLogDir)


def test_check_stage_contract_rejects_stage_with_no_method():
    class MissingMethod:
        stages = (StageSpec(name="train", config="training"),)

    with pytest.raises(StageContractError, match="defines no method"):
        check_stage_contract(MissingMethod)


def test_check_stage_contract_rejects_public_method_not_a_stage():
    class ExtraPublicMethod:
        stages = (StageSpec(name="train", config="training"),)

        def train(self):
            pass

        def helper(self):
            pass

    with pytest.raises(StageContractError, match="helper is public but not a stage"):
        check_stage_contract(ExtraPublicMethod)


def test_check_stage_contract_allows_private_helpers():
    class WithPrivateHelper:
        stages = (StageSpec(name="train", config="training"),)

        def train(self):
            pass

        def _helper(self):
            pass

    check_stage_contract(WithPrivateHelper)


def test_check_stage_contract_accepts_valid_declaration():
    class Good:
        stages = (
            StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
            StageSpec(name="pack_model", config="publication"),
        )

        def create_dataset(self):
            pass

        def pack_model(self):
            pass

    check_stage_contract(Good)


# ---------------------------------------------------------------------------
# check_requested_stages: request-time errors
# ---------------------------------------------------------------------------


class _ExampleSystem:
    stages = (
        StageSpec(name="train", config="training"),
        StageSpec(name="pack_model", config="publication"),
    )

    def train(self):
        pass

    def pack_model(self):
        pass


def test_check_requested_stages_rejects_unknown_stage():
    with pytest.raises(StageContractError, match=r"has no stage \['decode'\]"):
        check_requested_stages(_ExampleSystem, ["decode"], {"training": object()})


def test_check_requested_stages_rejects_missing_config():
    with pytest.raises(
        StageContractError, match="pack_model runs on the publication config"
    ):
        check_requested_stages(
            _ExampleSystem, ["train", "pack_model"], {"training": object()}
        )


def test_check_requested_stages_accepts_satisfied_request():
    check_requested_stages(
        _ExampleSystem,
        ["train", "pack_model"],
        {"training": object(), "publication": object()},
    )


# ---------------------------------------------------------------------------
# stage_names / stage_spec / stage_log_dir
# ---------------------------------------------------------------------------


def test_stage_names_preserves_declared_order():
    assert stage_names(_ExampleSystem) == ["train", "pack_model"]


def test_stage_spec_returns_the_matching_spec():
    assert stage_spec(_ExampleSystem, "pack_model").config == "publication"


def test_stage_spec_raises_for_unknown_name():
    with pytest.raises(StageContractError, match="has no stage 'decode'"):
        stage_spec(_ExampleSystem, "decode")


def test_stage_log_dir_reads_the_configured_path(tmp_path):
    class WithConfig:
        stages = (StageSpec(name="train", config="training", log_dir="exp_dir"),)
        training_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        _default_log_dir = tmp_path / "logs"

        def train(self):
            pass

    assert stage_log_dir(WithConfig(), "train") == tmp_path / "exp"


def test_stage_log_dir_falls_back_when_config_is_none():
    class NoConfig:
        stages = (StageSpec(name="train", config="training", log_dir="exp_dir"),)
        training_config = None
        _default_log_dir = Path("logs")

        def train(self):
            pass

    assert stage_log_dir(NoConfig(), "train") == Path("logs")


def test_stage_log_dir_falls_back_when_log_dir_is_none():
    class NoLogDir:
        stages = (StageSpec(name="pack_model", config="publication"),)
        publication_config = OmegaConf.create({"out_dir": "somewhere"})
        _default_log_dir = Path("logs")

        def pack_model(self):
            pass

    assert stage_log_dir(NoLogDir(), "pack_model") == Path("logs")


# ---------------------------------------------------------------------------
# roles: config roles derived from a system's own stages
# ---------------------------------------------------------------------------


def test_roles_preserves_first_seen_stage_order():
    assert roles(_ExampleSystem) == ("training", "publication")


def test_roles_drops_duplicates_keeping_first_occurrence():
    class RepeatingRoles:
        stages = (
            StageSpec(name="train", config="training"),
            StageSpec(name="collect_stats", config="training"),
            StageSpec(name="infer", config="inference"),
        )

        def train(self):
            pass

        def collect_stats(self):
            pass

        def infer(self):
            pass

    assert roles(RepeatingRoles) == ("training", "inference")


def test_roles_is_empty_for_a_system_with_no_stages_declared():
    class NoRoles:
        stages = ()

    assert roles(NoRoles) == ()
