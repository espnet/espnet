import pytest


@pytest.fixture(scope="session")
def tiny_bert_checkpoint(tmp_path_factory):
    """Build a local BERT checkpoint for tokenizer and SLU training tests."""
    transformers = pytest.importorskip("transformers")
    path = tmp_path_factory.mktemp("tiny-bert")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "increase",
        "the",
        "heating",
        "in",
        "bathroom",
        "a",
        "i",
    ]
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(str(vocab_file))
    tokenizer.save_pretrained(path)
    config = transformers.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
    )
    transformers.BertModel(config).save_pretrained(path)
    return str(path)
