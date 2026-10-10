"""A tiny, offline F5-TTS recipe for the template tests.

The recipe is laid out under ``egs3/<corpus_name>/f5tts/`` inside a
temporary directory, the way a real one is, with a freshly initialised model
saved as its checkpoint. Its configs are deltas over the template's defaults,
as a recipe's are; the training one is written out in full so the test reads
as one file. Nothing is trained and nothing is downloaded: the vocoder is
replaced by a stand-in.
"""

import shutil
import sys
from importlib import resources

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
from espnet3.systems.f5tts.f5tts import F5TTS
from espnet3.systems.f5tts.inference import F5TTSInference
from espnet3.utils.config_utils import load_and_merge_config

PACKAGE = "egs3.TEMPLATE.f5tts"
TOKENS = ["<blank>", "<unk>", "a", "b", "c", "<space>", "<sos/eos>"]
MODEL_OVERRIDES = {
    "hidden_size": 32,
    "depth": 1,
    "attention_heads": 2,
    "attention_head_size": 16,
    "feed_forward_multiplier": 1,
    "text_embedding_size": 16,
    "convolution_layers": 1,
}
FEATS_EXTRACT_CONFIG = {
    "fs": 24000,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 100,
}

# The recipe's training config: the shape `egs3/libritts/f5tts` uses, cut
# down to a model that builds in milliseconds and a 40000-step schedule. The
# data stages run in-process (`parallel: null`): a Dask cluster, which a real
# recipe's `parallel` block would start, takes longer than a unit test may.
TRAINING_CONFIG = {
    "num_device": 1,
    "num_nodes": 1,
    "task": None,
    "recipe_dir": ".",
    "data_dir": "${recipe_dir}/data",
    "exp_tag": "${self_name:}",
    "exp_dir": "${recipe_dir}/exp/${exp_tag}",
    "stats_dir": "${exp_dir}/stats",
    "inference_dir": "${exp_dir}/inference",
    "create_dataset": {"recipe_dir": "${recipe_dir}"},
    # `_recursive_: false` comes from the template, as it does for a recipe.
    "dataset": {
        "_target_": "espnet3.components.data.data_organizer.DataOrganizer",
        "recipe_dir": "${recipe_dir}",
        "train": [
            {
                "data_src_args": {
                    "manifest_path": "${remove_long_short.save_path}/train.tsv"
                }
            }
        ],
        "valid": [
            {
                "data_src_args": {
                    "manifest_path": "${remove_long_short.save_path}/valid.tsv"
                }
            }
        ],
        "preprocessor": {
            "_target_": "espnet2.train.preprocessor.CommonPreprocessor",
            "token_type": "${create_token_list.token_type}",
            "token_list": "${token_list}",
            "text_cleaner": "${create_token_list.cleaner}",
            "g2p_type": "${create_token_list.g2p}",
        },
    },
    "remove_long_short": {
        "min_wav_duration": 1.0,
        "max_wav_duration": 20.0,
        "splits": ["train", "valid"],
        "manifest_paths": {
            "train": "${data_dir}/manifest/train.tsv",
            "valid": "${data_dir}/manifest/valid.tsv",
        },
        "save_path": "${data_dir}/manifest_filtered",
    },
    "create_token_list": {
        "manifest_path": "${remove_long_short.save_path}/train.tsv",
        "save_path": "${data_dir}/token_list",
        "filename": "tokens.txt",
        "token_type": "char",
        "cleaner": None,
        "g2p": None,
        "add_symbol": ["<blank>:0", "<unk>:1", "<sos/eos>:-1"],
    },
    "token_list": "${create_token_list.save_path}/${create_token_list.filename}",
    "model": {
        "_target_": "espnet3.systems.f5tts.f5tts.F5TTS",
        "token_list": "${token_list}",
        "feats_extract_config": FEATS_EXTRACT_CONFIG,
        **MODEL_OVERRIDES,
    },
    "optimizer": {"_target_": "torch.optim.AdamW", "lr": 7.5e-5},
    "scheduler": {
        "_target_": (
            "espnet3.components.schedulers.linear_warmup_decay.LinearWarmupDecayLR"
        ),
        "warmup_steps": 2000,
        "total_steps": "${trainer.max_steps}",
    },
    "scheduler_interval": "step",
    "best_model_criterion": [["valid/loss", 1, "min"]],
    "parallel": None,
    "dataloader": {
        "collate_fn": {
            "_target_": "espnet2.train.collate_fn.CommonCollateFn",
            "int_pad_value": 0,
            "float_pad_value": 0.0,
        },
        "train": {
            "iter_factory": {
                "_target_": (
                    "espnet2.iterators.sequence_iter_factory.SequenceIterFactory"
                ),
                "shuffle": True,
                "collate_fn": "${dataloader.collate_fn}",
                "batches": {
                    "type": "numel",
                    "shape_files": ["${stats_dir}/train/feats_shape"],
                    "batch_size": 1,
                    "min_batch_size": 1,
                    "batch_bins": 3840000,
                },
            }
        },
        "valid": {
            "iter_factory": {
                "_target_": (
                    "espnet2.iterators.sequence_iter_factory.SequenceIterFactory"
                ),
                "shuffle": False,
                "collate_fn": "${dataloader.collate_fn}",
                "batches": {
                    "type": "numel",
                    "shape_files": ["${stats_dir}/valid/feats_shape"],
                    "batch_size": 1,
                    "min_batch_size": 1,
                    "batch_bins": 3840000,
                },
            }
        },
    },
    "trainer": {
        "accelerator": "auto",
        "devices": "${num_device}",
        "num_nodes": "${num_nodes}",
        "strategy": "auto",
        "max_steps": 40000,
        "callbacks": [
            {
                "_target_": "espnet3.components.callbacks.ema.EMACallback",
                "decay": 0.9999,
            }
        ],
        "logger": [
            {
                "_target_": "lightning.pytorch.loggers.TensorBoardLogger",
                "save_dir": "${exp_dir}/tensorboard",
                "name": "tb_logger",
            }
        ],
    },
    "fit": {},
}

# The template names the Inference and rebuilds it from `${exp_dir}/config.yaml`;
# a recipe adds the sampling settings.
INFERENCE_CONFIG = {
    "model": {
        "vocoder_path": None,
        "target_sample_rate": 24000,
        "ode_solver_steps": 2,
        "guidance_strength": 2.0,
        "sway_sampling_coefficient": -1.0,
        "speed": 1.0,
        "seed": 0,
    },
    "batch_size": None,
}

# The recipe's `dataset` package: what espnet3 imports when a split names no
# `data_src`. It reads the four-column manifest the data stages write.
TOY_DATASET = '''"""Toy LibriTTS-shaped dataset for the template tests."""

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Serve ``utt_id<TAB>wav_path<TAB>text<TAB>speaker`` rows."""

    def __init__(self, manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            self.rows = [line.rstrip("\\n").split("\\t") for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        _, wav_path, text, _ = self.rows[idx]
        speech, _ = sf.read(wav_path, dtype="float32")
        return {"text": text, "speech": np.asarray(speech, dtype=np.float32)}
'''

# A recipe overrides only what differs from the template: the directory its
# token list lives in (OmegaConf replaces the list, so the rest is restated).
PUBLICATION_CONFIG = {
    "pack_model": {"include": ["src", "${data_dir}/token_list"]},
}

# The demo wiring, README and requirements are the template's; a recipe adds
# an example row and the prompt recording it names.
DEMO_CONFIG = {
    "recipe_dir": ".",
    "ui": {"examples": [["a cab", "examples/prompt.wav", "abba"]]},
    "pack": {"include": ["examples/prompt.wav"]},
}


class StubVocos:
    """Stands in for Vocos: exposes ``decode``, upsamples by the hop length."""

    def decode(self, mel):
        """Return silence one hop length long per mel frame."""
        return torch.zeros(1, mel.shape[-1] * 256)


@pytest.fixture(autouse=True)
def isolated_process_state(monkeypatch):
    """Keep process-global state from leaking into or out of a test.

    A packed bundle ships a top-level ``src`` package and is imported by
    putting the bundle on ``sys.path``; a ``src`` already imported from
    another directory would be found instead. The parallel config is
    module-global too, and a multi-worker one left by another test would
    start a Dask cluster for the data stages run here.
    """
    monkeypatch.setattr(parallel_module, "parallel_config", None)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def forget_bundled_modules():
        """Drop any imported ``src`` or ``dataset`` package from ``sys.modules``."""
        for name in list(sys.modules):
            if name.split(".")[0] in ("src", "dataset"):
                del sys.modules[name]

    forget_bundled_modules()
    yield
    forget_bundled_modules()


@pytest.fixture
def stub_vocoder(monkeypatch):
    """Replace the Vocos download with :class:`StubVocos`."""
    monkeypatch.setattr(F5TTSInference, "_load_vocoder", lambda self, path: StubVocos())


def template_path(*parts):
    """Return the path of a file shipped in the template package."""
    return resources.files(PACKAGE).joinpath(*parts)


@pytest.fixture
def recipe_dir(tmp_path, monkeypatch):
    """A trained-looking recipe directory; the working directory is set to it."""
    recipe = tmp_path / "egs3" / "minicorpus" / "f5tts"
    (recipe / "conf").mkdir(parents=True)
    (recipe / "dataset").mkdir()
    (recipe / "dataset" / "__init__.py").write_text(TOY_DATASET, encoding="utf-8")

    # The example prompt `conf/demo.yaml` names, as `create_dataset` would leave it.
    (recipe / "examples").mkdir()
    sf.write(
        recipe / "examples" / "prompt.wav", np.zeros(12000, dtype=np.float32), 24000
    )

    # src/: the template's helpers, copied as its README tells a recipe to.
    (recipe / "src").mkdir()
    for name in ("__init__.py", "app.py"):
        shutil.copy(template_path("src", name), recipe / "src" / name)

    # conf/: the recipe's configs, merged by `run.py` over the template's
    # defaults.
    for name, content in {
        "training.yaml": TRAINING_CONFIG,
        "inference.yaml": INFERENCE_CONFIG,
        "publication.yaml": PUBLICATION_CONFIG,
        "demo.yaml": DEMO_CONFIG,
    }.items():
        OmegaConf.save(OmegaConf.create(content), recipe / "conf" / name)

    # What create_token_list and train would have written.
    token_list = recipe / "data" / "token_list" / "tokens.txt"
    token_list.parent.mkdir(parents=True)
    token_list.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    exp_dir = recipe / "exp" / "training"
    exp_dir.mkdir(parents=True)
    # What F5TTSSystem.train writes beside the checkpoint: the training config
    # as run.py loads and resolves it.
    training_config = load_and_merge_config(
        recipe / "conf" / "training.yaml", "training.yaml", default_package=PACKAGE
    )
    OmegaConf.save(training_config, exp_dir / "config.yaml")
    model = F5TTS(
        token_list=str(token_list),
        feats_extract_config=FEATS_EXTRACT_CONFIG,
        **MODEL_OVERRIDES,
    )
    torch.save({"state_dict": model.state_dict()}, exp_dir / "last.ckpt")
    # Files the template's exclude patterns must keep out of the bundle.
    (exp_dir / "step40000.ckpt").write_bytes(b"periodic checkpoint")
    (exp_dir / "train.log").write_text("log", encoding="utf-8")
    (exp_dir / "stats").mkdir()
    (exp_dir / "stats" / "feats_shape").write_text("utt 10,100\n", encoding="utf-8")

    monkeypatch.chdir(recipe)
    return recipe
