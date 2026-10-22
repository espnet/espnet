"""Exercise GeoLID orchestration with small stand-ins for shared pipeline stages."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[3] / "egs2/geolid/lid1"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Recipes use POSIX scripts")
CORPORA = ("voxlingua107", "fleurs", "ml_superb2", "babel", "voxpopuli")


@pytest.fixture
def recipe(tmp_path):
    """Record calls and require materialized dumps before filter/combine."""
    (tmp_path / "run.sh").symlink_to(RECIPE / "run.sh")
    (tmp_path / "local").mkdir()
    stub = tmp_path / "pipeline_stub.py"
    stub.write_text("""import json
import os
import sys
from pathlib import Path

step = sys.argv[1]
args = sys.argv[2:]
if args == ['--help']:
    print('Shared LID pipeline options')
    sys.exit(0)
options = {key.replace('-', '_'): value
           for key, value in zip(args[::2], args[1::2])}
with Path('calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'step': step, 'options': options}) + '\\n')
if os.environ.get('FAIL_STEP') == step:
    sys.exit(1)
if step == 'lid':
    if int(options.get('__stage', 1)) <= 3 <= int(options.get('__stop_stage', 10000)):
        dump = Path(options.get('__dumpdir', 'dump')) / options['__feats_type']
        splits = options['__test_sets'].split()
        skip_train = options.get('__skip_train', 'false') == 'true'
        if not skip_train or options.get('__eval_valid_set', 'false') == 'true':
            splits += options['__valid_set'].split()
        if not skip_train:
            splits.append(options['__train_set'])
        for split in splits:
            (dump / split).mkdir(parents=True, exist_ok=True)
elif step == 'filter':
    dump = Path(options['__dump_dir'])
    for split in options['__splits'].split():
        assert (dump / split).is_dir(), split
        (dump / split.replace('_lang', '_over_10s_lang')).mkdir(exist_ok=True)
elif step == 'combine':
    dump = Path(options['__dump_dir'])
    corpora = ('voxlingua107', 'fleurs', 'ml_superb2', 'babel_over_10s', 'voxpopuli')
    for corpus in corpora:
        assert (dump / f'train_{corpus}_lang').is_dir(), corpus
    Path(options['__train_out']).mkdir(parents=True, exist_ok=True)
""")
    for filename, step in (
        ("lid.sh", "lid"),
        ("local/filter_babel_over_10s.sh", "filter"),
        ("local/combine.sh", "combine"),
    ):
        script = tmp_path / filename
        script.write_text(
            "#!/usr/bin/env bash\n"
            f'exec {shlex.quote(sys.executable)} {shlex.quote(str(stub))} {step} "$@"\n'
        )
        script.chmod(0o755)
    return tmp_path


def run(recipe, *args, check=True):
    return subprocess.run(
        ["bash", "run.sh", *args],
        cwd=recipe,
        check=check,
        capture_output=True,
        text=True,
    )


def calls(recipe):
    return [
        json.loads(line) for line in (recipe / "calls.jsonl").read_text().splitlines()
    ]


def test_default_voxlingua107(recipe):
    run(recipe)
    (call,) = calls(recipe)
    assert call["step"] == "lid"
    assert call["options"]["__train_set"] == "train_voxlingua107_lang"
    assert call["options"]["__lid_config"] == "conf/train.yaml"
    assert call["options"]["__expdir"] == "exp_voxlingua107_only"


def test_combined_prepares_all_splits_before_training(recipe):
    run(recipe, "--setup", "combined")
    recorded = calls(recipe)
    assert [call["step"] for call in recorded] == ["lid"] * 5 + [
        "filter",
        "combine",
        "lid",
    ]
    for corpus, call in zip(CORPORA, recorded[:5]):
        options = call["options"]
        assert options["__stage"] == "1"
        assert options["__stop_stage"] == "3"
        assert options["__train_set"] == f"train_{corpus}_lang"
        assert options["__valid_set"] == f"dev_{corpus}_lang"
        assert options["__local_data_opts"].split() == ["--datasets", corpus]
    assert recorded[1]["options"]["__test_sets"] == "test_fleurs_lang"
    assert recorded[2]["options"]["__test_sets"] == "dev_dialect_ml_superb2_lang"
    assert recorded[4]["options"]["__test_sets"] == "test_voxpopuli_lang"
    options = recorded[-1]["options"]
    assert options["__stage"] == "4"
    assert options["__train_set"] == "train_all_no_filter_lang"
    assert options["__valid_set"] == "dev_ml_superb2_lang"
    assert options["__tsne_set"] == "train_all_no_filter_lang"
    assert options["__lid_config"] == "conf/tuning/combined.yaml"
    assert options["__expdir"] == "exp_combined"
    assert options["__test_sets"].split() == [
        "dev_voxlingua107_lang",
        "test_voxpopuli_lang",
        "test_fleurs_lang",
        "dev_ml_superb2_lang",
        "dev_dialect_ml_superb2_lang",
        "dev_babel_over_10s_lang",
    ]


@pytest.mark.parametrize("stage,stop", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 5)])
def test_combined_preparation_stage_boundaries(recipe, stage, stop):
    run(recipe, "--setup", "combined", "--stage", str(stage), "--stop-stage", str(stop))
    recorded = calls(recipe)
    expected = ["lid"] * 5
    if stop >= 3:
        expected += ["filter", "combine"]
    if stop >= 4:
        expected += ["lid"]
    assert [call["step"] for call in recorded] == expected
    for call in recorded[:5]:
        assert call["options"]["__stage"] == str(stage)
        assert call["options"]["__stop_stage"] == str(min(stop, 3))
    if stop >= 4:
        assert recorded[-1]["options"]["__stage"] == "4"
        assert recorded[-1]["options"]["__stop_stage"] == str(stop)


@pytest.mark.parametrize("stage,stop", [(4, 4), (4, 7), (5, 7), (6, 7)])
def test_combined_resume_does_not_prepare_data(recipe, stage, stop):
    run(recipe, "--setup", "combined", "--stage", str(stage), "--stop_stage", str(stop))
    (call,) = calls(recipe)
    assert call["step"] == "lid"
    assert call["options"]["__stage"] == str(stage)
    assert call["options"]["__stop_stage"] == str(stop)


def test_combined_preserves_overrides_and_passes_shared_options(recipe):
    overrides = {
        "train_set": "custom_train",
        "valid_set": "custom_valid",
        "test_sets": "test_one test_two",
        "tsne_set": "test_one",
        "lid_config": "conf/custom.yaml",
        "expdir": "custom_exp",
        "dumpdir": "custom_dump",
        "python": "coverage run --append",
        "nj": "2",
        "ngpu": "0",
        "local_data_opts": (
            "--babel_path /corpora/babel --voxpopuli_path /corpora/voxpopuli"
        ),
    }
    args = [item for key, value in overrides.items() for item in (f"--{key}", value)]
    run(recipe, "--setup", "combined", "--stop_stage", "7", *args)
    recorded = calls(recipe)
    for key, value in overrides.items():
        assert recorded[-1]["options"][f"__{key}"] == value
    for corpus, call in zip(CORPORA, recorded[:5]):
        assert call["options"]["__train_set"] == f"train_{corpus}_lang"
        assert call["options"]["__local_data_opts"] == (
            overrides["local_data_opts"] + f" --datasets {corpus}"
        )
    for call in recorded[5:7]:
        assert call["options"]["__dump_dir"] == "custom_dump/raw"
        assert call["options"]["__python"] == "coverage run --append"
    assert recorded[6]["options"]["__train_out"] == "custom_dump/raw/custom_train"


def test_skip_data_prep_still_formats_and_combines(recipe):
    run(recipe, "--setup", "combined", "--skip_data_prep", "true", "--stop_stage", "3")
    recorded = calls(recipe)
    assert [call["step"] for call in recorded] == ["lid"] * 5 + ["filter", "combine"]
    assert all(call["options"]["__skip_data_prep"] == "true" for call in recorded[:5])


@pytest.mark.parametrize("eval_valid_set", ["false", "true"])
def test_skip_train_only_filters_babel_dev(recipe, eval_valid_set):
    run(
        recipe,
        "--setup",
        "combined",
        "--skip_train",
        "true",
        "--eval_valid_set",
        eval_valid_set,
        "--stop_stage",
        "3",
    )
    recorded = calls(recipe)
    assert [call["step"] for call in recorded] == ["lid"] * 5 + ["filter"]
    assert recorded[-1]["options"]["__splits"] == "dev_babel_lang"
    for corpus in CORPORA:
        assert (recipe / "dump/raw" / f"dev_{corpus}_lang").is_dir()
        assert not (recipe / "dump/raw" / f"train_{corpus}_lang").exists()


def test_failed_preparation_does_not_start_training(recipe, monkeypatch):
    monkeypatch.setenv("FAIL_STEP", "filter")
    result = run(recipe, "--setup", "combined", check=False)
    assert result.returncode != 0
    assert [call["step"] for call in calls(recipe)] == ["lid"] * 5 + ["filter"]


@pytest.mark.parametrize("args", [("--setup", "invalid"), ("--setup",), ("--stage=3",)])
def test_invalid_arguments_fail_before_preparation(recipe, args):
    assert run(recipe, *args, check=False).returncode != 0
    assert not (recipe / "calls.jsonl").exists()


def test_help_includes_setup_and_shared_options(recipe):
    result = run(recipe, "--help")
    assert "--setup voxlingua107|combined" in result.stdout
    assert "Shared LID pipeline options" in result.stdout
    assert not (recipe / "calls.jsonl").exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--feats_type", "raw_copy"),
        ("--fs", "8k"),
        ("--speed_perturb_factors", "0.9 1.0"),
    ],
)
def test_combined_rejects_incompatible_waveform_options(recipe, option, value):
    result = run(recipe, "--setup", "combined", option, value, check=False)
    assert result.returncode != 0
    assert "requires --feats_type raw" in result.stderr
    assert not (recipe / "calls.jsonl").exists()


def test_shell_config_respects_resume_and_explicit_stage_override(recipe):
    (recipe / "run.conf").write_text("stage=4\nstop_stage=4\n")
    run(recipe, "--setup", "combined", "--config", "run.conf", "--stage", "5")
    (call,) = calls(recipe)
    assert call["step"] == "lid"
    assert call["options"]["__stage"] == "5"
    assert call["options"]["__stop_stage"] == "4"


def test_combined_validates_waveform_options_from_shell_config(recipe):
    (recipe / "run.conf").write_text("fs=8k\n")
    result = run(recipe, "--setup", "combined", "--config", "run.conf", check=False)
    assert result.returncode != 0
    assert "requires --feats_type raw" in result.stderr
    assert not (recipe / "calls.jsonl").exists()


def test_shell_config_preserves_dataset_and_experiment_choices(recipe):
    configured = {
        "train_set": "custom_train",
        "valid_set": "custom_valid",
        "test_sets": "custom_test another_test",
        "tsne_set": "custom_tsne",
        "lid_config": "custom.yaml",
        "expdir": "configured_exp",
    }
    (recipe / "run.conf").write_text(
        "stage=4\nstop_stage=4\n"
        + "".join(f"{key}={shlex.quote(value)}\n" for key, value in configured.items())
    )
    run(recipe, "--setup", "combined", "--config", "run.conf", "--expdir", "cli_exp")
    (call,) = calls(recipe)
    for key, value in configured.items():
        assert call["options"][f"__{key}"] == ("cli_exp" if key == "expdir" else value)
