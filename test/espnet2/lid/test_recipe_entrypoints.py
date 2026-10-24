"""Run the GeoLID shell entrypoints without downloading speech corpora."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[3] / "egs2/geolid/lid1"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Recipes use POSIX scripts")


@pytest.fixture
def recipe_environment(tmp_path, monkeypatch):
    """Use real shell utilities with isolated manifests and Python executables."""
    (tmp_path / "local").mkdir()
    (tmp_path / "utils").symlink_to(RECIPE / "utils", target_is_directory=True)
    for name in ("path.sh", "cmd.sh", "db.sh"):
        (tmp_path / name).touch()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("python", "python3"):
        executable = binaries / name
        executable.write_text(
            f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n'
        )
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    return tmp_path


def test_stage1_data_entrypoint(recipe_environment, monkeypatch):
    recipe = recipe_environment
    local = recipe / "local"
    (local / "data.sh").symlink_to(RECIPE / "local/data.sh")
    (local / "prepare_lang2vec.py").symlink_to(RECIPE / "local/prepare_lang2vec.py")
    # Replace only corpus acquisition and the external feature database.
    (local / "prepare_voxlingua107.sh").write_text(
        "set -eu\n"
        "mkdir -p data/train_voxlingua107_lang\n"
        "printf 'eng utterance\n' > data/train_voxlingua107_lang/lang2utt\n"
    )
    package = recipe / "lang2vec"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "lang2vec.py").write_text(
        "def get_features(languages, feature):\n"
        "    assert languages == ['eng'] and feature == 'geo'\n"
        "    return {'eng': [0.25, 0.5]}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(recipe))

    subprocess.run(["./local/data.sh"], cwd=recipe, check=True)

    assert (recipe / "data/voxlingua107_lang2vec.txt").read_text() == ("eng 0.25 0.5\n")


def test_stage6_ood_entrypoint(recipe_environment):
    recipe = recipe_environment
    for name in ("prepare_ood_test.sh", "prepare_ood_test.py"):
        (recipe / "local" / name).symlink_to(RECIPE / "local" / name)
    dump = recipe / "dump/raw"
    train = dump / "train_all_no_filter_lang"
    train.mkdir(parents=True)
    (train / "lang2utt").write_text("eng train_eng\nfra train_fra\n")
    test = dump / "test_fleurs_lang"
    test.mkdir()
    (test / "lang2utt").write_text("deu unseen\neng seen\n")
    (test / "utt2lang").write_text("seen eng\nunseen deu\n")
    (test / "wav.scp").write_text("seen /audio/seen.wav\nunseen /audio/unseen.wav\n")

    subprocess.run(
        [
            "./local/prepare_ood_test.sh",
            "--dump_dir",
            "dump/raw",
            "--train_set",
            train.name,
            "--test_sets",
            test.name,
        ],
        cwd=recipe,
        check=True,
    )

    result = dump / f"{test.name}_cross_{train.name}"
    assert (result / "utt2lang").read_text() == "seen eng\n"
    assert (result / "wav.scp").read_text() == "seen /audio/seen.wav\n"
    for name in ("lang2utt", "category2utt"):
        assert (result / name).read_text() == "eng seen\n"


@pytest.mark.parametrize("dataset", ["babel", "voxpopuli"])
def test_stage1_forwards_corpus_path(recipe_environment, dataset):
    recipe = recipe_environment
    (recipe / "local/data.sh").symlink_to(RECIPE / "local/data.sh")
    (recipe / "local" / f"prepare_{dataset}.sh").write_text(
        "set -eu\n"
        '[ "$1" = --dataset_path ]\n'
        'printf "%s\\n" "$2" > corpus_path.txt\n'
    )
    subprocess.run(
        [
            "./local/data.sh",
            "--datasets",
            dataset,
            f"--{dataset}_path",
            "/corpora/custom",
        ],
        cwd=recipe,
        check=True,
    )
    assert (recipe / "corpus_path.txt").read_text() == "/corpora/custom\n"


def test_combine_entrypoint_custom_dump_and_output(recipe_environment, monkeypatch):
    recipe = recipe_environment
    for name in ("combine.sh", "combine_lid_data.py", "prepare_lang2vec.py"):
        (recipe / "local" / name).symlink_to(RECIPE / "local" / name)
    dump = recipe / "custom_dump/raw"
    corpora = ("fleurs", "ml_superb2", "voxlingua107", "voxpopuli", "babel_over_10s")
    for corpus in corpora:
        source = dump / f"train_{corpus}_lang"
        source.mkdir(parents=True)
        (source / "wav.scp").write_text(f"{corpus}_utt /audio/{corpus}.wav\n")
        (source / "utt2lang").write_text(f"{corpus}_utt eng\n")
        (source / "utt2num_samples").write_text(f"{corpus}_utt 160000\n")
        (source / "audio_format").write_text("wav\n")
        (source / "feats_type").write_text("raw\n")
    package = recipe / "lang2vec"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "lang2vec.py").write_text(
        "def get_features(languages, feature):\n" "    return {'eng': [0.25, 0.5]}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(recipe))
    subprocess.run(
        [
            "bash",
            "local/combine.sh",
            "--dump_dir",
            "custom_dump/raw",
            "--train_out",
            "custom_dump/raw/custom_train",
        ],
        cwd=recipe,
        check=True,
    )
    output = dump / "custom_train"
    assert len((output / "wav.scp").read_text().splitlines()) == 5
    assert len((output / "dataset2utt").read_text().splitlines()) == 5
    assert (recipe / "data/combined_lang2vec.txt").read_text() == "eng 0.25 0.5\n"
