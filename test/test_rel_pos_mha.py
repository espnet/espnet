import torch

from espnet.nets.pytorch_backend.transformer.attention import RelPositionMultiHeadedAttention, \
    RelPositionMultiHeadedAttentionForPaddedSequences


def test_rel_position_multi_headed_attention_for_padded_sequences():
    attention_params = dict(
        n_head=4,
        n_feat=256,
        dropout_rate=0.1,
        zero_triu=False
    )
    rel_pos_mha = RelPositionMultiHeadedAttention(**attention_params)
    rel_pos_mha.eval()  # for reproducibility
    rel_pos_mha_for_padded = RelPositionMultiHeadedAttentionForPaddedSequences(**attention_params)
    rel_pos_mha_for_padded.eval()  # for reproducibility

    batch_size = 128
    max_len = 64
    def pad_sequence_nd(sequence):
        sizes = list(sequence[0].shape)
        for entry in sequence:
            for i in range(len(sizes)):
                if entry.shape[i] > sizes[i]:
                    sizes[i] = entry.shape[i]
        result = torch.zeros(len(sequence), *sizes)
        for i in range(len(sequence)):
            input_slice = [i]
            for j in sequence[i].shape:
                input_slice.append(slice(j))
            result[tuple(input_slice)] = sequence[i]
        return result

    sequence_lens = torch.randint(
        low=1,
        high=max_len,
        size=(batch_size,)
    )
    xs = [
        torch.randn(attention_params['n_head'], length, 2 * length - 1) for length in sequence_lens
    ]
    masks = [
        torch.ones(length, dtype=torch.int) for length in sequence_lens
    ]


    # showcase that RelPositionMultiHeadedAttention rel_shift is not consistent
    # with batches of padded sequences

    entry_wise_outputs = []
    for x in xs:
        x_shifted = rel_pos_mha.rel_shift(
            x.unsqueeze(0)  # creating fictional batch dimension
        )
        entry_wise_outputs.append(
            x_shifted.squeeze(0)  # removing fictional batch dimension
        )
    entry_wise_outputs_padded = pad_sequence_nd(entry_wise_outputs)

    batch_shifted = rel_pos_mha.rel_shift(pad_sequence_nd(xs))

    assert entry_wise_outputs_padded.shape == batch_shifted.shape
    assert not torch.allclose(entry_wise_outputs_padded, batch_shifted)

    # showcase that RelPositionalEncodingForPaddedSequences is consistent
    # with batches of padded sequences and entry-wise output of RelPositionalEncoding

    entry_wise_outputs_for_padded = []
    for x, mask in zip(xs, masks):
        x_shifted = rel_pos_mha_for_padded.rel_shift(
            x.unsqueeze(0),  # creating fictional batch dimension
            mask.unsqueeze(0)  # creating fictional batch dimension
        )
        entry_wise_outputs_for_padded.append(
            x_shifted.squeeze(0)  # removing fictional batch dimension
        )
    entry_wise_outputs_padded2 = pad_sequence_nd(entry_wise_outputs)

    batch_shifted_for_padded = rel_pos_mha_for_padded.rel_shift(
        pad_sequence_nd(xs),
        torch.nn.utils.rnn.pad_sequence(
            masks,
            batch_first=True,
            padding_value=0
        )
    )
    assert entry_wise_outputs_padded.shape == entry_wise_outputs_padded2.shape
    assert torch.allclose(entry_wise_outputs_padded, entry_wise_outputs_padded2)
    assert torch.allclose(batch_shifted_for_padded, entry_wise_outputs_padded2)
    assert torch.allclose(entry_wise_outputs_padded, batch_shifted_for_padded)
