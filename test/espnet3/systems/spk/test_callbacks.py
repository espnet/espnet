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
