import torch

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)
from espnet2.tok.tokenizer import SpeechTokenizer
from espnet2.train.tok_gan_trainer import SpeechTokenizerGANTrainer


class DummyFrontend(AbsFrontend):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 2)

    def output_size(self):
        return 2

    def forward(self, input, input_lengths):
        return self.projection(input.unsqueeze(-1)), input_lengths


class DummyQuantizer(AbsSpeechTokenizerQuantizer):
    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.centroids = torch.nn.Parameter(torch.randn(2, 2))

    @property
    def num_clusters(self):
        return 2

    def _output(self, features, lengths):
        logits = features @ self.centroids.transpose(0, 1)
        assignment = logits.softmax(dim=-1)
        token_ids = assignment.argmax(dim=-1)
        return SpeechTokenizerOutput(
            continuous=features,
            assignment=assignment,
            token_ids=token_ids,
            lengths=lengths,
            soft_assignment=assignment,
        )

    def forward(self, features, feature_lengths):
        return self._output(features, feature_lengths)

    def encode(self, features, feature_lengths):
        return self._output(features, feature_lengths)


def test_prepare_epoch_can_reset_main_optimizer():
    tokenizer = SpeechTokenizer(DummyFrontend(), DummyQuantizer(), freeze_epochs=1)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1.0e-3)
    tokenizer.set_epoch(2)
    loss = tokenizer(torch.randn(1, 2), torch.tensor([2])).assignment.square().sum()
    loss.backward()
    optimizer.step()
    assert len(optimizer.state) > 0

    tokenizer.set_epoch(1)
    just_unfroze = SpeechTokenizerGANTrainer.prepare_epoch(
        tokenizer,
        [optimizer],
        epoch=2,
        reset_optimizer_on_unfreeze=True,
    )

    assert just_unfroze
    assert len(optimizer.state) == 0


def test_prepare_epoch_can_reset_both_optimizers():
    model = SpeechTokenizer(DummyFrontend(), DummyQuantizer(), freeze_epochs=1)
    main = torch.optim.Adam(model.parameters(), lr=1e-3)
    discriminator_param = torch.nn.Parameter(torch.ones(()))
    discriminator = torch.optim.Adam([discriminator_param], lr=1e-3)
    for optimizer, parameter in (
        (main, next(model.parameters())),
        (discriminator, discriminator_param),
    ):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert len(optimizer.state) > 0

    model.set_epoch(1)

    just_unfroze = SpeechTokenizerGANTrainer.prepare_epoch(
        model=model,
        optimizers=[main, discriminator],
        epoch=2,
        reset_optimizer_on_unfreeze=True,
    )

    assert just_unfroze
    assert len(main.state) == 0
    assert len(discriminator.state) == 0
