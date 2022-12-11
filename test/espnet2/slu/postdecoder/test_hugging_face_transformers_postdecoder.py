import pytest
import torch
from packaging.version import parse as V

from espnet2.slu.postdecoder.hugging_face_transformers_postdecoder import (
    HuggingFaceTransformersPostDecoder,
)

is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")


@pytest.mark.execution_timeout(50)
def test_transformers_forward():
    if not is_torch_1_8_plus:
        return
    postdecoder = HuggingFaceTransformersPostDecoder("bert-base-cased", 400)
    max_length = 128
    transcript_data = ["increase the heating in the bathroom"]
    (
        transcript_input_id_features,
        transcript_input_mask_features,
        transcript_segment_ids_feature,
        transcript_position_ids_feature,
        input_id_length,
    ) = postdecoder.convert_examples_to_features(transcript_data, max_length)
    y = postdecoder(
        torch.LongTensor(transcript_input_id_features),
        torch.LongTensor(transcript_input_mask_features),
        torch.LongTensor(transcript_segment_ids_feature),
        torch.LongTensor(transcript_position_ids_feature),
    )
    odim = postdecoder.output_size()
    assert y.shape == torch.Size([1, max_length, odim])


@pytest.mark.execution_timeout(30)
def test_convert_examples_to_features():
    if not is_torch_1_8_plus:
        return
    postdecoder = HuggingFaceTransformersPostDecoder("bert-base-cased", 400)
    max_length = 128
    transcript_data = ["increase the heating in the bathroom"]
    (
        transcript_input_id_features,
        transcript_input_mask_features,
        transcript_segment_ids_feature,
        transcript_position_ids_feature,
        input_id_length,
    ) = postdecoder.convert_examples_to_features(transcript_data, max_length)
    assert torch.LongTensor(transcript_input_id_features).shape == torch.Size(
        [1, max_length]
    )
    assert torch.LongTensor(transcript_input_mask_features).shape == torch.Size(
        [1, max_length]
    )
    assert torch.LongTensor(transcript_segment_ids_feature).shape == torch.Size(
        [1, max_length]
    )
    assert torch.LongTensor(transcript_position_ids_feature).shape == torch.Size(
        [1, max_length]
    )
    assert torch.LongTensor(input_id_length).shape == torch.Size([1])
