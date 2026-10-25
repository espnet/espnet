import pytest
import torch

from espnet2.aqa.metric_tokenizer.metric_tokenizer import MetricTokenizer


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


@pytest.mark.parametrize(
    "options, token",
    [
        ({}, 13),
        (dict(categorical_scheme="index"), 13),
        (dict(categorical_scheme="legacy"), 12),
    ],
)
def test_categorical_scheme_defaults_to_index(options, token):
    """Omitting the scheme keeps index tokens; legacy bins the last label."""
    tokenizer = MetricTokenizer(dict(token_info(), **options), ["mos", "language"])
    assert tokenizer.metric2token({"mos": 1.5, "language": "jpn"}) == {
        "mos": (4, 7),
        "language": (10, token),
    }


@pytest.mark.parametrize(
    "category, token, decoded",
    [("eng", 12, "eng"), ("jpn", 13, "jpn"), ("zho", 13, "jpn")],
)
def test_legacy_category_round_trip(legacy_token_info, category, token, decoded):
    """Legacy labels keep their threshold bins, so the last two share a token."""
    tokenizer = MetricTokenizer(legacy_token_info, None)
    metrics = {"mos": 1.5, "language": category, "bg_event": "music"}
    pairs = tokenizer.metric2token(metrics)
    assert pairs == {"mos": (4, 7), "language": (10, token), "bg_event": (15, 18)}
    sequence = [2] + [t for pair in pairs.values() for t in pair]
    assert tokenizer.tokenseq2metric(sequence, True) == {
        "mos": [1.0],
        "language": [decoded],
        "bg_event": ["music"],
    }
    reduced = tokenizer.metric2token(metrics, reduce_offset=True)
    for name, (_, value) in pairs.items():
        assert tokenizer.add_offset([reduced[name][1]], name) == [value]
    # Bins outside the trained range still decode, as numerical bins do.
    assert [tokenizer.token2metric(t, "language") for t in (11, 14)] == ["eng", "zho"]


@pytest.mark.parametrize(
    "shifted, tokens", [(["bg_event"], [17, 18, 19]), ([], [18, 16, 18])]
)
def test_shifted_categorical_override(legacy_token_info, shifted, tokens):
    """Listed metrics use index tokens while other legacy metrics keep bins."""
    legacy_token_info["shifted_categorical_metrics"] = shifted
    tokenizer = MetricTokenizer(legacy_token_info, None)
    labels = legacy_token_info["tokenizer"]["bg_event"]
    pairs = [tokenizer.metric2token({"bg_event": label}) for label in labels]
    assert pairs == [{"bg_event": (15, token)} for token in tokens]
    assert tokenizer.metric2token({"language": "zho"}) == {"language": (10, 13)}
    if shifted:
        assert [tokenizer.token2metric(t, "bg_event") for t in tokens] == labels
        with pytest.raises(ValueError, match="Invalid token value"):
            tokenizer.token2metric(16, "bg_event")


@pytest.mark.parametrize("scheme", ["index", "legacy"])
@pytest.mark.parametrize("metric", ["language", "bg_event"])
def test_unknown_category_raises_unless_skipped(legacy_token_info, scheme, metric):
    """Unknown labels raise by default and drop only that metric when opted in."""
    legacy_token_info["categorical_scheme"] = scheme
    metrics = {"mos": 1.5, metric: "fra"}
    with pytest.raises(ValueError, match="Invalid category value: fra"):
        MetricTokenizer(legacy_token_info, None).metric2token(metrics)
    tokenizer = MetricTokenizer(legacy_token_info, None, skip_unknown_categories=True)
    assert tokenizer.metric2token(metrics) == {"mos": (4, 7)}


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(categorical_scheme="shifted"), "categorical_scheme: shifted"),
        (dict(shifted_categorical_metrics=["mos"]), "mos is not"),
        (dict(shifted_categorical_metrics=["lang"]), "lang is not"),
    ],
)
def test_reject_invalid_categorical_options(options, message):
    """Reject unknown schemes and overrides that name no categorical metric."""
    with pytest.raises(ValueError, match=message):
        MetricTokenizer(dict(token_info(), **options), None)
