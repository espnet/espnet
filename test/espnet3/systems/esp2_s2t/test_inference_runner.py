"""Tests for the SOT runner and renderer in espnet3/systems/esp2_s2t/src/."""

import numpy as np
import pytest

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.esp2_s2t.src.inference_runner import ESP2S2TInferenceRunner

PROMPT = ["<|en|>", "<|transcribe|>"]
# The contract of egs3/TEMPLATE/esp2_s2t/conf/inference.yaml.
CONTRACT = {
    "inputs": {"speech": {"kind": "audio", "sample_rate": 16000, "channels": 1}},
    "outputs": {"hyp": {"kind": "text"}},
}


def _result(*tokens):
    """Return a result in Speech2Text's n-best format, with special tokens kept."""
    tokens = PROMPT + list(tokens)
    return [("".join(tokens), tokens, list(range(len(tokens))), None, None)]


@pytest.mark.parametrize("separator", ["????", "<sc>"])
def test_the_hypothesis_keeps_its_timestamps_and_separator(separator):
    """format_output keeps the timestamps and the separator in the hypothesis.

    DER reads the timestamps, and both metrics split the text on the separator.
    """
    result = _result(
        "<|0.00|>", " hi", "<|1.20|>", f" {separator} ", "<|0.50|>", " yes", "<|0.90|>"
    )
    assert ESP2S2TInferenceRunner.format_output(result, batched=False) == {
        "hyp": f"<|0.00|> hi<|1.20|> {separator} <|0.50|> yes<|0.90|>"
    }


def test_a_hypothesis_rendered_without_special_tokens_is_refused():
    """format_output raises ValueError for a hypothesis without special tokens.

    A converter that drops special tokens loses the prompt and the timestamps.
    Scoring such a text would treat every group as no speech, so the runner raises
    an error here.
    """
    text_only = [(" hi yes", [" hi", " yes"], [1, 2], " hi yes", None)]
    with pytest.raises(ValueError, match="prompt"):
        ESP2S2TInferenceRunner.format_output(text_only, batched=False)


def test_no_hypothesis_is_an_empty_hyp():
    assert ESP2S2TInferenceRunner.format_output([], batched=False) == {"hyp": ""}


def test_a_batch_is_refused():
    """format_output raises ValueError for batched output.

    The timestamp rules use one prompt length for the whole beam.
    """
    with pytest.raises(ValueError, match="batch_size: null"):
        ESP2S2TInferenceRunner.format_output([_result()], batched=True)


def test_forward_resamples_to_the_contract_rate_and_keeps_the_items_id():
    """The forward call resamples the speech to the contract rate and keeps the id.

    The call goes through the shared runner. It applies the input contract, renders
    the hypothesis and applies the output contract.
    """
    seen = {}

    def model(speech):
        seen["speech"] = speech
        return _result("<|0.00|>", " hi", "<|0.40|>")

    dataset = [{"utt_id": "EN2002a-0.37-11.83", "speech": (8000, np.zeros(8000))}]
    output = ESP2S2TInferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key="speech",
        inference_contract=load_inference_contract(CONTRACT),
    )

    assert output == {"hyp": "<|0.00|> hi<|0.40|>", "utt_id": "EN2002a-0.37-11.83"}
    assert isinstance(seen["speech"], np.ndarray)
    assert seen["speech"].shape == (16000,)


def test_forward_refuses_audio_with_more_than_one_channel():
    dataset = [{"utt_id": "u1", "speech": Audio(np.zeros((2, 16000)), 16000)}]
    with pytest.raises(Exception, match="channel"):
        ESP2S2TInferenceRunner.forward(
            0,
            dataset=dataset,
            model=lambda speech: _result(),
            input_key="speech",
            inference_contract=load_inference_contract(CONTRACT),
        )
