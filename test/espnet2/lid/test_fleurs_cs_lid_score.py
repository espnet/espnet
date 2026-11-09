"""Test egs2/fleurs_cs_lid/lid1/local/score.py (LID language-set scoring)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)
from espnet2.train.preprocessor import LIDPreprocessor

SCRIPT = Path(__file__).resolve().parents[3] / "egs2/fleurs_cs_lid/lid1/local/score.py"
_spec = importlib.util.spec_from_file_location("fleurs_cs_lid_score", SCRIPT)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

LANGS = ["ara", "cmn", "deu", "eng", "fra"]


def write(path, text):
    """Write ``text`` to ``path`` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def loss_logits(loss, embds):
    """Logits of the loss module, captured at the input of its cross entropy."""
    captured = []
    hook = loss.ce.register_forward_hook(lambda m, i, o: captured.append(i[0]))
    loss(embds, torch.zeros(len(embds), dtype=torch.long))
    hook.remove()
    return captured[0].detach().numpy()


def test_logits_and_topk_match_loss_module():
    """Logits and top-k equal those of the margin-free training loss."""
    torch.manual_seed(0)
    num_classes, num_sub, dim = len(LANGS), 3, 8
    # margin 0 and k_top 0: the training logits are scale * max sub-center cosine
    loss = ArcMarginProduct_intertopk_subcenter(
        dim, num_classes, scale=32.0, margin=0.0, K=num_sub, k_top=0
    )
    embds = torch.randn(6, dim)
    expected = loss_logits(loss, embds)
    logits = score.class_logits(embds, loss.weight.detach(), num_classes, loss.scale)
    np.testing.assert_allclose(logits, expected, rtol=1e-5, atol=1e-5)
    _, _, preds = loss(embds)
    np.testing.assert_array_equal(logits.argmax(1), preds.numpy())
    for k in (1, 2, 3):
        ref = frozenset(LANGS[:k])
        for row, expected_row in zip(logits, expected):
            top = torch.topk(torch.from_numpy(expected_row), k).indices
            assert score.predict_set(row, LANGS, ref) == {LANGS[i] for i in top}


def test_sub_center_layout():
    """Weight rows are (C * K, D) grouped per class: row c * K + k."""
    num_classes, num_sub, dim = 3, 2, 3
    weight = torch.zeros(num_classes * num_sub, dim)
    weight[:, 1] = 1.0
    weight[1 * num_sub + 1] = torch.tensor([1.0, 0.0, 0.0])  # class 1, sub-center 1
    embds = torch.tensor([[1.0, 0.0, 0.0]])
    logits = score.class_logits(embds, weight, num_classes, 10.0)
    np.testing.assert_allclose(logits, [[0.0, 10.0, 0.0]], atol=1e-6)
    loss = ArcMarginProduct_intertopk_subcenter(
        dim, num_classes, scale=10.0, margin=0.0, K=num_sub, k_top=0
    )
    loss.weight.data.copy_(weight)
    np.testing.assert_allclose(logits, loss_logits(loss, embds), atol=1e-6)


def test_label_order_matches_training_mapping(tmp_path):
    """Class index i is line i of lang2utt, as LIDPreprocessor maps it."""
    lang2utt = write(tmp_path / "lang2utt", "eng u1 u2\nara-eng u3\ncmn u4\n")
    labels = list(score.read_kv(lang2utt))
    mapping = LIDPreprocessor(train=False, lang2utt=str(lang2utt)).lang2label
    assert {label: i for i, label in enumerate(labels)} == mapping


def test_k_is_the_reference_size():
    """The predicted set size is taken from the reference: k = len(ref)."""
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert score.predict_set(logits, LANGS, {"eng"}) == {"ara"}
    assert score.predict_set(logits, LANGS, {"deu", "eng"}) == {"ara", "cmn"}
    assert score.predict_set(logits, LANGS, {"deu", "eng", "fra"}) == {
        "ara",
        "cmn",
        "deu",
    }


def test_atomic_pair_classes_use_top1():
    """Pair classes take the top-1 class; independent ones the top k."""
    logits = np.array([1.0, 3.0, 2.0])
    atomic = ["ara-eng", "cmn-eng", "eng"]
    independent = ["ara", "cmn", "eng"]
    assert score.predict_set(logits, atomic, {"cmn", "eng"}) == {"cmn", "eng"}
    assert score.predict_set(logits, atomic, {"eng"}) == {"cmn", "eng"}
    assert score.predict_set(logits, independent, {"ara", "eng"}) == {"cmn", "eng"}
    # A 3-language reference is always wrong with pair classes
    three = {"ara", "cmn", "eng"}
    assert score.predict_set(logits, atomic, three) != three
    assert score.predict_set(logits, independent, three) == three


def test_ties_go_to_the_lower_class_index():
    """Equal logits rank the lower class index first."""
    logits = np.array([1.0, 2.0, 2.0, 2.0, 0.0])
    assert score.predict_set(logits, LANGS, {"eng"}) == {"cmn"}
    assert score.predict_set(logits, LANGS, {"ara", "eng"}) == {"cmn", "deu"}


@pytest.mark.parametrize(
    "text, message",
    [
        ("u1 eng\n\n", "utt2langs:2: expected"),
        ("u1 eng\nu2\n", "utt2langs:2: expected"),
        ("u1 eng\nu2 eng eng\n", "utt2langs:2: repeated language"),
        ("u1 eng\nu2 ara-eng ara\n", "utt2langs:2: repeated language"),
        ("u1 eng\nu1 ara\n", "utt2langs:2: duplicate"),
    ],
)
def test_malformed_references(tmp_path, text, message):
    """Malformed utt2langs lines are errors with file:line."""
    with pytest.raises(ValueError, match=message):
        score.read_refs(write(tmp_path / "utt2langs", text))


def test_blank_line_in_lang2utt(tmp_path):
    """A blank lang2utt line is an error with file:line."""
    with pytest.raises(ValueError, match="lang2utt:1: expected"):
        score.read_kv(write(tmp_path / "lang2utt", "\neng u1\n"))


@pytest.fixture
def lid_exp(tmp_path):
    """Build a model directory as lid.sh stage 6 leaves it (K = 2 sub-centers).

    Classes (lang2utt order): eng, ara, cmn. Weight rows are one-hot, so an
    embedding 2 * e[c * 2] + e[c2 * 2 + 1] ranks class c first and c2 second.
    """
    exp = tmp_path / "exp"
    write(exp / "config.yaml", yaml.safe_dump({"loss_conf": {"scale": 16}}))
    torch.save({"loss.weight": torch.eye(6)}, exp / "valid.accuracy.best.pth")
    infer = exp / "inference/valid.accuracy.best/test"
    infer.mkdir(parents=True)

    def embd(first, second):
        x = np.zeros(6, dtype=np.float32)
        x[2 * first], x[2 * second + 1] = 2.0, 1.0
        return x

    embds = {"u1": embd(0, 2), "u2": embd(0, 1), "u3": embd(0, 1)}
    write(tmp_path / "data/test/utt2langs", "u1 eng\nu2 ara eng\nu3 cmn eng\n")
    return {
        "infer": infer,
        "embds": embds,
        "argv": [
            "--pred_lids",
            str(infer / "test_lids"),
            "--target_lids",
            str(tmp_path / "data/test/utt2lang"),
            "--train_lang2utt",
            str(write(tmp_path / "lang2utt", "eng t1\nara t2\ncmn t3\n")),
            "--seen_utt2langs",
            str(write(tmp_path / "seen", "s1 eng ara\ns2 eng\n")),
            "--results",
            str(tmp_path / "out/results"),
        ],
    }


def save_embds(lid_exp, embds):
    """Write the embeddings that stage 6 would extract."""
    np.savez(lid_exp["infer"] / "test_utt_embds.npz", **embds)


def test_main_writes_results_and_details(lid_exp, tmp_path):
    """Score one set, then merge its details into the same results."""
    save_embds(lid_exp, lid_exp["embds"])
    score.main(lid_exp["argv"])
    out = tmp_path / "out"
    results = (out / "results").read_text()
    # u2 (ara eng) is seen although the seen file lists it as "eng ara"
    assert results == (
        "# Exact language-set accuracy (%)\ngroup\tn\taccuracy\n"
        "all\t3\t66.67\n1-language\t1\t100.00\n2-language\t2\t50.00\n"
        "seen\t1\t100.00\nunseen\t1\t0.00\n"
    )
    assert (out / score.DETAILS).read_text() == (
        "utt\tref\tpred\texact\n"
        "u1\teng\teng\t1\nu2\tara eng\tara eng\t1\nu3\tcmn eng\tara eng\t0\n"
    )
    seen = lid_exp["argv"][lid_exp["argv"].index("--seen_utt2langs") + 1]
    merged = tmp_path / "all/results"
    argv = ["--details", str(out / score.DETAILS), "--seen_utt2langs", seen]
    score.main(argv + ["--results", str(merged)])
    assert merged.read_text() == results


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e.pop("u3"), "1 references without embedding"),
        (lambda e: e.update(u4=e["u1"]), "1 embeddings without reference"),
    ],
)
def test_embeddings_must_match_references(lid_exp, change, message):
    """A missing or extra embedding is an error."""
    embds = dict(lid_exp["embds"])
    change(embds)
    save_embds(lid_exp, embds)
    with pytest.raises(ValueError, match=message):
        score.main(lid_exp["argv"])


def test_malformed_details_line(tmp_path):
    """A malformed --details line is an error with file:line."""
    details = write(tmp_path / score.DETAILS, "utt\tref\tpred\texact\nu1\teng\n")
    with pytest.raises(ValueError, match="lang_set_details:2: malformed"):
        score.read_details([details])


def test_duplicate_details_ids(tmp_path):
    """An utterance repeated within or across --details files is an error."""
    header = "utt\tref\tpred\texact\n"
    row = "u1\teng\teng\t1\n"
    one = write(tmp_path / "a" / score.DETAILS, header + row + row)
    with pytest.raises(ValueError, match="lang_set_details:3: malformed or duplicate"):
        score.read_details([one])
    a = write(tmp_path / "a" / score.DETAILS, header + row)
    b = write(tmp_path / "b" / score.DETAILS, header + row)
    with pytest.raises(ValueError, match="lang_set_details:2: malformed or duplicate"):
        score.read_details([a, b])
