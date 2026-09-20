from types import SimpleNamespace

import torch

from espnet3.systems.spk.callbacks import SpeakerVerificationScoring


class _FakeModel:
    """Stand-in for the buffer interface of the speaker verification model."""

    def __init__(self):
        self.trial_scores = []
        self.trial_labels = []

    def reset_trials(self):
        self.trial_scores.clear()
        self.trial_labels.clear()

    def pop_trials(self):
        if not self.trial_scores:
            empty = torch.zeros(0)
            return empty, empty.long()
        scores = torch.cat(self.trial_scores).float()
        labels = torch.cat(self.trial_labels).long()
        self.reset_trials()
        return scores, labels


class _FakeModule:
    """Minimal LightningModule surface used by the callback."""

    def __init__(self):
        self.model = _FakeModel()
        self.logged = {}
        self.device = torch.device("cpu")

    def all_gather(self, tensor):
        return tensor

    def log_dict(self, values, **_kwargs):
        self.logged.update({key: float(value) for key, value in values.items()})


def _feed(callback, module, trainer, batches):
    callback.on_validation_epoch_start(trainer, module)
    for batch_idx, (scores, labels) in enumerate(batches):
        module.model.trial_scores.append(scores)
        module.model.trial_labels.append(labels)
        callback.on_validation_batch_end(trainer, module, None, None, batch_idx)


def test_metrics_are_logged_only_after_the_last_batch():
    callback = SpeakerVerificationScoring()
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=[2])

    callback.on_validation_epoch_start(trainer, module)
    module.model.trial_scores.append(torch.tensor([0.9, 0.2]))
    module.model.trial_labels.append(torch.tensor([1, 0]))
    callback.on_validation_batch_end(trainer, module, None, None, 0)
    assert module.logged == {}

    module.model.trial_scores.append(torch.tensor([0.8, 0.1]))
    module.model.trial_labels.append(torch.tensor([1, 0]))
    callback.on_validation_batch_end(trainer, module, None, None, 1)

    assert module.logged == {"valid/eer": 0.0, "valid/mindcf": 0.0}
    assert module.model.trial_scores == []


def test_epoch_start_drops_stale_trials():
    callback = SpeakerVerificationScoring()
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=[1])

    module.model.trial_scores.append(torch.tensor([0.5]))
    module.model.trial_labels.append(torch.tensor([1]))
    callback.on_validation_epoch_start(trainer, module)

    assert module.model.trial_scores == []


def test_single_class_batches_are_not_scored():
    callback = SpeakerVerificationScoring()
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=[1])

    _feed(callback, module, trainer, [(torch.tensor([0.5, 0.6]), torch.tensor([1, 1]))])

    assert module.logged == {}


def test_unsized_dataloaders_are_skipped():
    callback = SpeakerVerificationScoring()
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=float("inf"))

    _feed(callback, module, trainer, [(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))])

    assert module.logged == {}


def test_min_dcf_operating_point_is_configurable():
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=[1])
    batches = [(torch.tensor([0.5, 0.4, 0.6, 0.3]), torch.tensor([1, 1, 0, 0]))]

    _feed(SpeakerVerificationScoring(p_target=0.5), module, trainer, batches)
    lenient = module.logged["valid/mindcf"]

    module = _FakeModule()
    _feed(SpeakerVerificationScoring(p_target=0.01), module, trainer, batches)

    assert module.logged["valid/mindcf"] > lenient


class _FakeDDPModule(_FakeModule):
    """Two-rank stand-in whose peer holds a different number of trials.

    ``all_gather`` is called three times per reduction -- the trial counts,
    then the padded scores, then the padded labels -- so the peer contribution
    is built per call, the way a real collective would return one row per rank.
    """

    def __init__(self, peer_scores, peer_labels):
        super().__init__()
        self.peer_scores = peer_scores
        self.peer_labels = peer_labels
        self._calls = 0

    def all_gather(self, tensor):
        self._calls += 1
        if self._calls == 1:
            peer = tensor.new_tensor([self.peer_scores.numel()])
        else:
            source = self.peer_scores if self._calls == 2 else self.peer_labels
            peer = torch.zeros_like(tensor)
            peer[: source.numel()] = source
        return torch.stack([tensor, peer])


def test_ranks_with_unequal_trial_counts_are_gathered():
    callback = SpeakerVerificationScoring()
    module = _FakeDDPModule(torch.tensor([0.1]), torch.tensor([0]))
    trainer = SimpleNamespace(num_val_batches=[1])

    # 3 trials locally against 1 on the peer: `all_gather` needs one shape, so
    # the buffers have to be padded to 3 and unpadded again after the gather.
    _feed(
        callback,
        module,
        trainer,
        [(torch.tensor([0.9, 0.8, 0.2]), torch.tensor([1, 1, 0]))],
    )

    # All four trials separate perfectly; padding left in place would not.
    assert module.logged == {"valid/eer": 0.0, "valid/mindcf": 0.0}


def test_a_rank_without_trials_still_joins_the_collective():
    callback = SpeakerVerificationScoring()
    module = _FakeDDPModule(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    trainer = SimpleNamespace(num_val_batches=[1])

    # This rank saw no trial batch. Returning early here would leave the peer
    # waiting in `all_gather` forever, so it must reach the collective anyway
    # and score the peer's trials.
    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(trainer, module, None, None, 0)

    assert module.logged == {"valid/eer": 0.0, "valid/mindcf": 0.0}
