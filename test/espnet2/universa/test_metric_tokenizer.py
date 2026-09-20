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
