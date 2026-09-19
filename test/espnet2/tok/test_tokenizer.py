import torch

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)
from espnet2.tok.tokenizer import SpeechTokenizer


class DummyFrontend(AbsFrontend):
    def output_size(self):
        return 2

    def forward(self, input, input_lengths):
        return input.unsqueeze(-1).repeat(1, 1, 2), input_lengths


class DummyQuantizer(AbsSpeechTokenizerQuantizer):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.forward_count = 0
        self.encode_count = 0

    @property
    def num_clusters(self):
        return 2

    def _output(self, features, lengths):
        token_ids = (features[..., 0] > 0).long()
        assignment = torch.nn.functional.one_hot(token_ids, 2).float()
        return SpeechTokenizerOutput(
            continuous=features,
            assignment=assignment,
            token_ids=token_ids,
            lengths=lengths,
            soft_assignment=assignment,
        )

    def forward(self, features, feature_lengths):
        self.forward_count += 1
        return self._output(features, feature_lengths)

    def encode(self, features, feature_lengths):
        self.encode_count += 1
        return self._output(features, feature_lengths)


def test_tokenizer_composes_frontend_and_quantizer():
    tokenizer = SpeechTokenizer(DummyFrontend(), DummyQuantizer())
    speech = torch.tensor([[-1.0, 2.0]])
    lengths = torch.tensor([2])

    output = tokenizer(speech, lengths)

    assert output.continuous.shape == (1, 2, 2)
    assert output.token_ids.tolist() == [[0, 1]]
    assert tokenizer.num_clusters == 2


def test_tokenizer_freezes_then_unfreezes_by_epoch():
    quantizer = DummyQuantizer()
    tokenizer = SpeechTokenizer(DummyFrontend(), quantizer, freeze_epochs=2)
    speech = torch.tensor([[-1.0, 2.0]])
    lengths = torch.tensor([2])

    tokenizer.train()
    tokenizer.set_epoch(1)
    tokenizer(speech, lengths)
    assert tokenizer.is_frozen
    assert quantizer.encode_count == 1
    assert quantizer.forward_count == 0

    assert tokenizer.set_epoch(2) is False
    tokenizer(speech, lengths)
    assert quantizer.encode_count == 2

    assert tokenizer.set_epoch(3) is True
    tokenizer(speech, lengths)
    assert not tokenizer.is_frozen
    assert quantizer.forward_count == 1

    tokenizer.eval()
    tokenizer(speech, lengths)
    assert quantizer.encode_count == 3
