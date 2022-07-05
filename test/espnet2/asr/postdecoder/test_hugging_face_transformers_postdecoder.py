import pytest
import torch

from espnet2.asr.postdecoder.hugging_face_transformers_postdecoder import (
    HuggingFaceTransformersPostDecoder,
)


# @pytest.mark.parametrize(
#     "model_name_or_path",
#     [
#         "akreal/tiny-random-bert",
#         "akreal/tiny-random-gpt2",
#         "akreal/tiny-random-xlnet",
#         "akreal/tiny-random-t5",
#         "akreal/tiny-random-mbart",
#         "akreal/tiny-random-mpnet",
#     ],
# )
@pytest.mark.execution_timeout(50)
def test_transformers_forward():
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

    # saved_param = postencoder.parameters().__next__().detach().clone()

    # postencoder.parameters().__next__().data *= 0
    # new_param = postencoder.parameters().__next__().detach().clone()
    # assert not torch.equal(saved_param, new_param)

    # postencoder.reload_pretrained_parameters()
    # new_param = postencoder.parameters().__next__().detach().clone()
    # assert torch.equal(saved_param, new_param)
