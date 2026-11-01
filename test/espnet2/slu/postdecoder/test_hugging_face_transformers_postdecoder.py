import pytest
import torch

from espnet2.slu.postdecoder.hugging_face_transformers_postdecoder import (
    HuggingFaceTransformersPostDecoder,
)


@pytest.mark.execution_timeout(50)
def test_transformers_forward(tiny_bert_checkpoint):
    postdecoder = HuggingFaceTransformersPostDecoder(tiny_bert_checkpoint, 400)
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
def test_convert_examples_to_features(tiny_bert_checkpoint):
    postdecoder = HuggingFaceTransformersPostDecoder(tiny_bert_checkpoint, 400)
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
    expected_ids = postdecoder.tokenizer.convert_tokens_to_ids(
        ["[CLS]", "increase", "the", "heating", "in", "the", "bathroom", "[SEP]"]
    )
    assert postdecoder.tokenizer.unk_token_id not in expected_ids
    assert input_id_length == [8]
    assert transcript_input_id_features == [expected_ids + [0] * (max_length - 8)]
    assert transcript_input_mask_features == [[1] * 8 + [0] * (max_length - 8)]
