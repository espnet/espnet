import pytest
import torch

from espnet2.sds.llm.hugging_face_llm import HuggingFaceLLM

pytest.importorskip("transformers")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("tag", ["HuggingFaceTB/SmolLM2-1.7B-Instruct"])
def test_forward(tag):
    llm = HuggingFaceLLM(tag=tag, access_token="")
    llm.warmup()
    x = [
        {
            "role": "system",
            "content": (
                "You are a helpful and friendly AI "
                "assistant. "
                "You are polite, respectful, and aim to "
                "provide concise and complete responses of "
                "less than 15 words."
            ),
        },
        {
            "role": "user",
            "content": ("Write me poem about Machine Learning."),
        },
    ]
    llm.forward(x)


WORDS = [
    "<unk>",
    "<eos>",
    "<bos>",
    "nothink",
    "hello",
    "world",
    "poem",
    "about",
    "speech",
]
# The template writes "nothink" when the caller sets enable_thinking to False,
# so a test can see whether chat_template_kwargs reached the template.
# The template writes <bos> itself, as Llama 3's does, and the tokenizer adds
# <bos> to the text it tokenizes, so a prompt tokenized twice has two of them.
TEMPLATE = (
    "<bos>{% if enable_thinking is defined and not enable_thinking %}nothink "
    "{% endif %}{% for m in messages %}{{ m['content'] }} {% endfor %}"
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Save a randomly initialized tiny Llama and a word-level tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny_llm")
    backend = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", WORDS.index("<bos>"))]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(path)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        eos_token_id=1,
        pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def _llm(path, **kwargs):
    return HuggingFaceLLM(tag=path, device="cpu", dtype="float32", **kwargs)


CHAT = [{"role": "user", "content": "hello world"}]


def test_access_token_is_optional(tiny_model_dir):
    _llm(tiny_model_dir).warmup()


@pytest.mark.parametrize(
    "kwargs, has_nothink",
    [({}, True), ({"chat_template_kwargs": {"enable_thinking": True}}, False)],
)
def test_chat_template_kwargs_reach_the_template(tiny_model_dir, kwargs, has_nothink):
    llm = _llm(tiny_model_dir, **kwargs)
    prompt = llm.tokenizer.decode(llm._encode(CHAT)["input_ids"][0])
    assert ("nothink" in prompt.split()) == has_nothink


def test_stream_matches_forward(tiny_model_dir):
    llm = _llm(tiny_model_dir, max_new_tokens=8)
    assert "".join(llm.stream(CHAT)).strip() == llm.forward(CHAT)


def test_stream_raises_the_generate_error(tiny_model_dir, monkeypatch):
    llm = _llm(tiny_model_dir)

    def fail(**kwargs):
        raise RuntimeError("generate failed")

    monkeypatch.setattr(llm.model, "generate", fail)
    with pytest.raises(RuntimeError, match="generate failed"):
        list(llm.stream(CHAT))


@pytest.mark.parametrize(
    "dtype, expected", [("float16", torch.float16), ("auto", None)]
)
def test_dtype_names_and_auto(tiny_model_dir, dtype, expected):
    llm = HuggingFaceLLM(tag=tiny_model_dir, device="cpu", dtype=dtype)
    # "auto" keeps the dtype the checkpoint was saved in, float32 here.
    assert llm.model.dtype == (expected or torch.float32)


def test_forward_with_return_dict_in_generate(tiny_model_dir):
    llm = _llm(tiny_model_dir, max_new_tokens=4)
    expected = llm.forward(CHAT)
    llm.model.generation_config.return_dict_in_generate = True
    assert llm.forward(CHAT) == expected


def test_prompt_has_one_bos(tiny_model_dir):
    llm = _llm(tiny_model_dir)
    ids = llm._encode(CHAT)["input_ids"][0].tolist()
    assert ids.count(WORDS.index("<bos>")) == 1
