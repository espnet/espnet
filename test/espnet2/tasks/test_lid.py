import numpy as np
import pytest
import torch

from espnet2.tasks.lid import LIDTask


def test_add_arguments():
    LIDTask.get_parser()


def test_add_arguments_help():
    parser = LIDTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        LIDTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        LIDTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        LIDTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        LIDTask.print_config(f)
    parser = LIDTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


@pytest.mark.parametrize(
    "preprocessor,loss_name",
    [
        ("lid", "aamsoftmax_sc_topk"),
        ("lid_softlabel", "aamsoftmax_sc_topk_softtarget"),
        ("lid_multilabel", "aamsoftmax_sc_topk_multilabel"),
    ],
)
def test_lid_target_preprocessor_and_loss_train_a_step(
    tmp_path, preprocessor, loss_name
):
    """Each LID preprocessor/loss pair builds, collates and trains a step."""
    lang2utt = tmp_path / "lang2utt"
    lang2utt.write_text("ara u1\neng u2\njpn u3\n")
    args = LIDTask.get_parser().parse_args([])
    args.lang2utt = str(lang2utt)
    args.lang_num = 3
    args.preprocessor = preprocessor
    args.preprocessor_conf = dict(
        fix_duration=False, noise_apply_prob=0.0, rir_apply_prob=0.0
    )
    args.frontend = None
    args.input_size = 7
    args.encoder = "identity"
    args.pooling = "mean"
    args.projector = "rawnet3"
    args.projector_conf = dict(output_size=4)
    args.loss = loss_name
    args.loss_conf = dict(k_top=0)
    preprocess = LIDTask.build_preprocess_fn(args, train=False)
    data = [
        ("u1", dict(speech=np.ones((5, 7), dtype=np.float32), lid_labels="ara")),
        (
            "u2",
            dict(
                speech=np.ones((5, 7), dtype=np.float32),
                lid_labels="eng" if preprocessor == "lid" else "eng jpn",
            ),
        ),
    ]
    _, batch = LIDTask.build_collate_fn(args, train=False)(
        [(uid, preprocess(uid, item)) for uid, item in data]
    )
    assert batch["lid_labels"].shape == (2, 1 if preprocessor == "lid" else 3)
    assert "lid_labels_lengths" not in batch
    model = LIDTask.build_model(args).eval()
    loss, _, _ = model(**batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
