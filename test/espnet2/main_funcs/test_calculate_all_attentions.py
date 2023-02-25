from collections import defaultdict

import numpy as np
import pytest
import torch

from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.rnn.attentions import AttAdd
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention, \
    RelPositionMultiHeadedAttentionForPaddedSequences


class Dummy(AbsESPnetModel):
    def __init__(self):
        super().__init__()
        self.att1 = MultiHeadedAttention(2, 10, 0.0)
        self.att2 = AttAdd(10, 20, 15)
        self.desired = defaultdict(list)

    def forward(self, x, x_lengths, y, y_lengths):
        self.att1(y, x, x, None)
        _, a2 = self.att2(x, x_lengths, y, None)
        self.desired["att1"].append(self.att1.attn.squeeze(0))
        self.desired["att2"].append(a2)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


class Dummy2(AbsESPnetModel):
    def __init__(self, atype):
        super().__init__()
        self.decoder = RNNDecoder(50, 128, att_conf=dict(atype=atype))

    def forward(self, x, x_lengths, y, y_lengths):
        self.decoder(x, x_lengths, y, y_lengths)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


def test_calculate_all_attentions_MultiHeadedAttention():
    model = Dummy()
    bs = 2
    batch = {
        "x": torch.randn(bs, 3, 10),
        "x_lengths": torch.tensor([3, 2], dtype=torch.long),
        "y": torch.randn(bs, 2, 10),
        "y_lengths": torch.tensor([4, 4], dtype=torch.long),
    }
    t = calculate_all_attentions(model, batch)
    print(t)
    for k in model.desired:
        for i in range(bs):
            np.testing.assert_array_equal(t[k][i].numpy(), model.desired[k][i].numpy())


@pytest.mark.parametrize(
    "atype",
    [
        "noatt",
        "dot",
        "add",
        "location",
        "location2d",
        "location_recurrent",
        "coverage",
        "coverage_location",
        "multi_head_dot",
        "multi_head_add",
        "multi_head_loc",
        "multi_head_multi_res_loc",
    ],
)
def test_calculate_all_attentions(atype):
    model = Dummy2(atype)
    bs = 2
    batch = {
        "x": torch.randn(bs, 20, 128),
        "x_lengths": torch.tensor([20, 17], dtype=torch.long),
        "y": torch.randint(0, 50, [bs, 7]),
        "y_lengths": torch.tensor([7, 5], dtype=torch.long),
    }
    t = calculate_all_attentions(model, batch)
    for k, o in t.items():
        for i, att in enumerate(o):
            print(att.shape)
            if att.dim() == 2:
                att = att[None]
            for a in att:
                assert a.shape == (batch["y_lengths"][i], batch["x_lengths"][i])


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
