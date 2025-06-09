# Copyright 2024 Masao Someki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import shutil
import tempfile
from pathlib import Path

import pytest
from torch import nn

import espnetez as ez
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr_transducer import ASRTransducerTask
from espnet2.tasks.asvspoof import ASVSpoofTask
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
from espnet2.tasks.gan_svs import GANSVSTask
from espnet2.tasks.gan_tts import GANTTSTask
from espnet2.tasks.hubert import HubertTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.mt import MTTask
from espnet2.tasks.s2st import S2STTask
from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.slu import SLUTask
from espnet2.tasks.spk import SpeakerTask
from espnet2.tasks.st import STTask
from espnet2.tasks.svs import SVSTask
from espnet2.tasks.tts import TTSTask
from espnet2.tasks.uasr import UASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

# Prepare directory in the test environment to store dummy files
TASK_CLASSES = [
    ("asr", ASRTask),
    ("asr_transducer", ASRTransducerTask),
    ("asvspoof", ASVSpoofTask),
    ("diar", DiarizationTask),
    ("enh_s2t", EnhS2TTask),
    ("enh_tse", TargetSpeakerExtractionTask),
    ("enh", EnhancementTask),
    ("gan_svs", GANSVSTask),
    ("gan_tts", GANTTSTask),
    ("hubert", HubertTask),
    ("lm", LMTask),
    ("mt", MTTask),
    ("s2st", S2STTask),
    ("s2t", S2TTask),
    ("slu", SLUTask),
    ("spk", SpeakerTask),
    ("st", STTask),
    ("svs", SVSTask),
    ("tts", TTSTask),
    ("uasr", UASRTask),
]


def generate_random_dataset(count: int):
    return [
        {
            "speech": f"/path/to/{i}.wav",
            "text": f"Lorem ipsum dolor sit amet {i}.",
        }
        for i in range(count)
    ]


def test_create_dump_file():
    with tempfile.TemporaryDirectory() as dump_dir:
        dump_dir = Path(dump_dir)
        dataset = generate_random_dataset(5)
        data_inputs = {
            "speech": ["wav.scp", "sound"],
            "text": ["text", "text"],
        }
        ez.data.create_dump_file(dump_dir, dataset, data_inputs)
        assert dump_dir.glob("wav.scp")
        assert dump_dir.glob("text")

        # check dump files
        with open(dump_dir / "wav.scp", "r") as f:
            for idx_l, line in enumerate(f.readlines()):
                line = line.strip().split()
                assert len(line) == 2
                assert line[1] == dataset[idx_l]["speech"]

        with open(dump_dir / "text", "r") as f:
            for idx_l, line in enumerate(f.readlines()):
                line = line.strip().split(maxsplit=1)
                assert line[1] == dataset[idx_l]["text"]


def test_join_dumps():
    dump_dir1 = tempfile.mktemp()
    dump_dir2 = tempfile.mktemp()
    dump_dir_out = tempfile.mktemp()

    try:
        dump_dir_out = Path(dump_dir_out)
        data_inputs = {
            "speech": ["wav.scp", "sound"],
            "text": ["text", "text"],
        }
        dataset1 = generate_random_dataset(5)
        ez.data.create_dump_file(dump_dir1, dataset1, data_inputs)

        dataset2 = generate_random_dataset(5)
        ez.data.create_dump_file(dump_dir2, dataset2, data_inputs)

        ez.data.join_dumps([dump_dir1, dump_dir2], ["dump_1", "dump_2"], dump_dir_out)

        assert dump_dir_out.glob("wav.scp")
        assert dump_dir_out.glob("text")

        concat_dataset = dataset1 + dataset2

        # check dump files
        with open(dump_dir_out / "wav.scp", "r") as f:
            for idx_l, line in enumerate(f.readlines()):
                line = line.strip().split()
                assert len(line) == 2
                assert line[1] == concat_dataset[idx_l]["speech"]

        with open(dump_dir_out / "text", "r") as f:
            for idx_l, line in enumerate(f.readlines()):
                line = line.strip().split(maxsplit=1)
                assert line[1] == concat_dataset[idx_l]["text"]

    finally:
        shutil.rmtree(dump_dir1)
        shutil.rmtree(dump_dir2)
        shutil.rmtree(dump_dir_out)


@pytest.mark.parametrize("task_name,task_class", TASK_CLASSES)
def test_task(task_name, task_class):
    task = ez.task.get_ez_task(task_name)
    assert issubclass(task, task_class)


@pytest.mark.parametrize("task_name,task_class", TASK_CLASSES)
def test_task_with_dataset(task_name, task_class):
    task = ez.task.get_ez_task_with_dataset(task_name)
    assert issubclass(task, task_class)


@pytest.mark.parametrize(
    "tr_dump,val_dump,tr_ds,val_ds,tr_dl,val_dl,test_case",
    [
        ["not None string" if i == "1" else None for i in format(idx_case, "06b")]
        + [format(idx_case, "06b")]
        for idx_case in range(64)
    ],
)
def test_check_argument(tr_dump, val_dump, tr_ds, val_ds, tr_dl, val_dl, test_case):
    print(test_case)
    try:
        ez.trainer.check_argument(
            train_dump_dir=tr_dump,
            valid_dump_dir=val_dump,
            train_dataset=tr_ds,
            valid_dataset=val_ds,
            train_dataloader=tr_dl,
            valid_dataloader=val_dl,
        )
        assert test_case in ("110000", "001100", "000011")
    except ValueError:
        assert test_case not in ("110000", "001100", "000011")


@pytest.mark.parametrize("task_name,task_class", TASK_CLASSES)
def test_load_config(task_name, task_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        # first create demo config
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text("""task: {task_name}""")
        default_config = task_class.get_default_config()
        ez_config = ez.config.from_yaml(task_name, config_path)

        for k in default_config.keys():
            assert default_config[k] == ez_config[k]


@pytest.mark.parametrize("task_name,task_class", TASK_CLASSES)
def test_update_finetune_config(task_name, task_class):
    with tempfile.TemporaryDirectory() as temp_dir:
        # first create demo config
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text("""use_lora: true""")
        pretrain_config = task_class.get_default_config()
        ez_config = ez.config.update_finetune_config(
            task_name, pretrain_config, config_path
        )

        for k, v in ez_config.items():
            if k != "use_lora":
                assert v == pretrain_config[k]
            else:
                assert v


def test_sentencepiece_preparation():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dataset = generate_random_dataset(5)
        data_inputs = {
            "speech": ["wav.scp", "sound"],
            "text": ["text", "text"],
        }
        ez.data.create_dump_file(temp_dir / "dump", dataset, data_inputs)
        ez.preprocess.prepare_sentences([temp_dir / "dump" / "text"], temp_dir / "spm")

        # It seems training an sentencepiece in CI and pytest may cause error.
        # So skip this test for now.
        vocab_size = 30
        model_type = "bpe"
        ez.preprocess.train_sentencepiece(
            temp_dir / "spm" / "train.txt",
            temp_dir / "data" / "bpemodel",
            vocab_size=vocab_size,
            model_type=model_type,
        )
        tokenizer = build_tokenizer(
            token_type=model_type,
            bpemodel=temp_dir / "data" / "bpemodel" / f"{model_type}.model",
        )
        token_id_converter = TokenIDConverter(
            token_list=temp_dir / "data" / "bpemodel" / "tokens.txt",
            unk_symbol="<unk>",
        )
        embedding = nn.Embedding(vocab_size, 2)
        ez.preprocess.add_special_tokens(
            tokenizer, token_id_converter, embedding, ["<ctc>", "<chat>"], "<unk>"
        )


def test_tokenizer_run():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        dataset = generate_random_dataset(5)
        data_inputs = {
            "speech": ["wav.scp", "sound"],
            "text": ["text", "text"],
        }
        ez.data.create_dump_file(temp_dir / "dump", dataset, data_inputs)
        ez.preprocess.prepare_sentences([temp_dir / "dump" / "text"], temp_dir / "spm")
        ez.preprocess.tokenize(
            str(temp_dir / "spm" / "train.txt"),
            str(temp_dir / "data" / "bpemodel"),
        )


def test_streaming_iter():
    from espnetez.dataloader import Dataloader

    dataset = generate_random_dataset(5)
    Dataloader(dataset=dataset).build_iter(0)


def test_dataset():
    dataset = generate_random_dataset(5)
    data_inputs = {
        "speech": lambda x: x,
        "text": lambda x: x,
    }
    ds = ez.ESPnetEZDataset(dataset, data_inputs)
    print(len(ds))
    print(ds.names())
    print(ds.has_name("speech"))
    print(ds[0])
