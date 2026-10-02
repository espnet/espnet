import datetime
from types import SimpleNamespace

import torch

from espnet3.systems.spk.callbacks import SpeakerVerificationScoring
from espnet3.systems.spk.scoring import TrialScores


class _FakeModel:
    """Stand-in for the trial interface of the speaker verification model."""

    def __init__(self):
        self.trial_metric = TrialScores()


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
        module.model.trial_metric.update(scores, labels)
        callback.on_validation_batch_end(trainer, module, None, None, batch_idx)


def test_metrics_are_logged_only_after_the_last_batch():
    callback = SpeakerVerificationScoring()
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=[2])

    callback.on_validation_epoch_start(trainer, module)
    module.model.trial_metric.update(torch.tensor([0.9, 0.2]), torch.tensor([1, 0]))
    callback.on_validation_batch_end(trainer, module, None, None, 0)
    assert module.logged == {}

    module.model.trial_metric.update(torch.tensor([0.8, 0.1]), torch.tensor([1, 0]))
    callback.on_validation_batch_end(trainer, module, None, None, 1)

    assert module.logged == {"valid/eer": 0.0, "valid/mindcf": 0.0}
    assert module.model.trial_metric.scores == []


def test_epoch_start_drops_stale_trials():
    callback = SpeakerVerificationScoring()
    module = _FakeModule()
    trainer = SimpleNamespace(num_val_batches=[1])

    module.model.trial_metric.update(torch.tensor([0.5]), torch.tensor([1]))
    callback.on_validation_epoch_start(trainer, module)

    assert module.model.trial_metric.scores == []


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


def _run_rank(rank, world_size, init_file, trials, results):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        # A rank that skips the collective makes its peers fail here instead
        # of hanging the test run.
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        callback = SpeakerVerificationScoring()
        module = _FakeModule()
        trainer = SimpleNamespace(num_val_batches=[1])
        callback.on_validation_epoch_start(trainer, module)
        for scores, labels in trials[rank]:
            module.model.trial_metric.update(scores, labels)
        callback.on_validation_batch_end(trainer, module, None, None, 0)
        results[rank] = dict(module.logged)
    finally:
        torch.distributed.destroy_process_group()


def _score_across_ranks(tmp_path, trials):
    ctx = torch.multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        results = manager.dict()
        torch.multiprocessing.start_processes(
            _run_rank,
            args=(len(trials), str(tmp_path / "init"), trials, results),
            nprocs=len(trials),
            start_method="spawn",
        )
        return [results[rank] for rank in range(len(trials))]


def test_ranks_with_unequal_trial_counts_are_gathered(tmp_path):
    # 3 trials against 1: every rank must score all four, and padding left
    # behind by the gather would break the perfect separation.
    logged = _score_across_ranks(
        tmp_path,
        [
            [(torch.tensor([0.9, 0.8, 0.2]), torch.tensor([1, 1, 0]))],
            [(torch.tensor([0.1]), torch.tensor([0]))],
        ],
    )

    assert logged == [{"valid/eer": 0.0, "valid/mindcf": 0.0}] * 2


def test_a_rank_without_trials_still_joins_the_collective(tmp_path):
    # Rank 1 saw no trial batch. If it skipped the gather, rank 0 would time
    # out waiting for it; instead both ranks score rank 0's trials.
    logged = _score_across_ranks(
        tmp_path,
        [[(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))], []],
    )

    assert logged == [{"valid/eer": 0.0, "valid/mindcf": 0.0}] * 2


def test_no_trials_on_any_rank_logs_nothing(tmp_path):
    logged = _score_across_ranks(tmp_path, [[], []])

    assert logged == [{}, {}]
