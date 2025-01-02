import pytest
import torch

from espnet2.sds.llm.hugging_face_llm import HuggingFaceLLM

pytest.importorskip("transformers")


@pytest.mark.parametrize("tag", ["HuggingFaceTB/SmolLM2-1.7B-Instruct"])
def test_forward(tag):
    if not torch.cuda.is_available():
        return  # Only GPU supported
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
