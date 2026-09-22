import pytest
import torch

from espnet2.universa.metric_tokenizer.metric_tokenizer import MetricTokenizer


def token_info():
    vocab, offset = [], {}
    tokenizers = {"mos": [0.0, 1.0, 2.0, 3.0], "language": ["eng", "jpn"]}
    for key, values in tokenizers.items():
        offset[key] = [len(vocab), len(values) + 2]
        vocab += [key + "@meta_label"] + [f"{key}@{i}" for i in range(len(values) + 1)]
    return dict(tokenizer=tokenizers, VOCAB=vocab, offset=offset)


def test_tokenizer_legacy_ids_and_categories():
    tokenizer = MetricTokenizer(token_info(), ["mos", "language"])
    # Numeric bins keep the historical strict upper-bound convention.
    pairs = tokenizer.metric2token({"mos": 1.5, "language": "jpn"})
    assert pairs["mos"] == (4, 7)
    assert pairs["language"] == (10, 13)
    decoded = tokenizer.tokenseq2metric([2, 4, 7, 10, 13], return_dict=True)
    assert decoded == {"mos": [1.0], "language": ["jpn"]}
    assert tokenizer.token2metric(13, "language") == "jpn"


def test_reduced_offsets_round_trip():
    tokenizer = MetricTokenizer(token_info(), ["mos", "language"])
    metrics = {"mos": 1.5, "language": "jpn"}
    reduced = tokenizer.metric2token(metrics, reduce_offset=True)
    full = tokenizer.metric2token(metrics)
    for name in metrics:
        assert tokenizer.add_offset([reduced[name][1]], name) == [full[name][1]]


def test_decoders_reject_mismatched_metric():
    tokenizer = MetricTokenizer(token_info(), ["mos", "language"])
    with pytest.raises(ValueError, match="mos"):
        tokenizer.token2metric(13, "mos")
    with pytest.raises(ValueError, match="mos"):
        tokenizer.tokenseq2metric([4, 13], return_dict=True)


def test_decoders_agree_on_numeric_zero_bin():
    tokenizer = MetricTokenizer(token_info(), ["mos"])
    label, value = tokenizer.metric2token({"mos": -1.0})["mos"]
    assert tokenizer.token2metric(value, "mos") == 0.0
    assert tokenizer.tokenseq2metric([label, value], True) == {"mos": [0.0]}


def test_decoders_reject_invalid_tokens():
    tokenizer = MetricTokenizer(token_info(), ["mos", "language"])
    for value in [-1, 0, 1, 2, 3, 10, 11, 14]:
        with pytest.raises(ValueError):
            tokenizer.tokenseq2metric([10, value], True)
    with pytest.raises(ValueError):
        tokenizer.tokenseq2metric([-4, 13], True)


@pytest.mark.parametrize(
    "value, index", [(-1.0, 0), (0.0, 1), (1.0, 2), (2.0, 3), (3.0, 3), (99.0, 3)]
)
def test_numeric_boundaries_keep_legacy_ids(value, index):
    tokenizer = MetricTokenizer(token_info(), ["mos"])
    assert tokenizer.metric2token({"mos": value}) == {"mos": (4, 5 + index)}


@pytest.mark.parametrize("category, token", [("eng", 12), ("jpn", 13)])
def test_category_round_trip(category, token):
    tokenizer = MetricTokenizer(token_info(), ["language"])
    assert tokenizer.metric2token({"language": category}) == {"language": (10, token)}
    assert tokenizer.token2metric(token, "language") == category
    assert tokenizer.tokenseq2metric([2, 10, token], True) == {"language": [category]}


@pytest.mark.parametrize("sequence", [list, iter, torch.tensor])
def test_decode_integer_sequences(sequence):
    tokenizer = MetricTokenizer(token_info(), ["mos", "language"])
    assert tokenizer.tokenseq2metric(sequence([2, 4, 7, 10, 13]), True) == {
        "mos": [1.0],
        "language": ["jpn"],
    }


def test_known_unselected_metric_is_skipped():
    tokenizer = MetricTokenizer(token_info(), ["mos"])
    assert tokenizer.metric2token({"mos": 1.5, "language": "eng"}) == {"mos": (4, 7)}


@pytest.mark.parametrize("selected", [["mos"], None])
def test_unknown_input_metric_is_rejected(selected):
    tokenizer = MetricTokenizer(token_info(), selected)
    with pytest.raises(ValueError, match="Unknown metric: mso"):
        tokenizer.metric2token({"mos": 1.5, "mso": 1.5})


def test_unknown_selected_metric_is_rejected():
    with pytest.raises(ValueError, match="Unknown selected metrics.*mso"):
        MetricTokenizer(token_info(), ["mso"])
