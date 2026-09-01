import pytest

import espnet3.components.data.iterator as iterator_module
from espnet3.components.data.iterator import EpochSyncIterator


def _patch_no_distributed(monkeypatch):
    monkeypatch.setattr(iterator_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        iterator_module.torch.distributed, "is_initialized", lambda: False
    )


def _patch_dist_gloo(monkeypatch, all_reduce):
    monkeypatch.setattr(iterator_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        iterator_module.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        iterator_module.torch.distributed, "get_backend", lambda: "gloo"
    )
    monkeypatch.setattr(iterator_module.torch.distributed, "all_reduce", all_reduce)


# --- Non-distributed passthrough ---


def test_yields_all_batches_without_distributed(monkeypatch):
    _patch_no_distributed(monkeypatch)

    def unexpected_all_reduce(tensor, op=None):
        raise AssertionError("all_reduce must not be called without distributed")

    monkeypatch.setattr(
        iterator_module.torch.distributed, "all_reduce", unexpected_all_reduce
    )
    iterator = EpochSyncIterator(["a", "b", "c"])
    assert list(iterator) == ["a", "b", "c"]


def test_len_delegates_to_wrapped_iterator():
    assert len(EpochSyncIterator([1, 2, 3])) == 3


def test_len_raises_when_wrapped_iterator_is_unsized():
    iterator = EpochSyncIterator(iter([1, 2, 3]))
    with pytest.raises(TypeError):
        len(iterator)


# --- Distributed epoch-end synchronization ---


def test_sync_yields_all_batches_when_ranks_aligned(monkeypatch):
    seen = []

    def fake_all_reduce(tensor, op=None):
        seen.append(tensor.item())

    _patch_dist_gloo(monkeypatch, fake_all_reduce)
    iterator = EpochSyncIterator(["a", "b", "c"])
    assert list(iterator) == ["a", "b", "c"]
    # 3 has-next=1 flags plus the final has-next=0 on local exhaustion
    assert seen == [1.0, 1.0, 1.0, 0.0]


def test_sync_stops_when_another_rank_is_exhausted(monkeypatch):
    calls = {"n": 0}

    def fake_all_reduce(tensor, op=None):
        calls["n"] += 1
        if calls["n"] >= 3:  # simulated other rank runs out at 3rd batch
            tensor.zero_()

    _patch_dist_gloo(monkeypatch, fake_all_reduce)
    iterator = EpochSyncIterator(["a", "b", "c"])
    assert list(iterator) == ["a", "b"]


def test_sync_iterator_is_reiterable(monkeypatch):
    def fake_all_reduce(tensor, op=None):
        pass

    _patch_dist_gloo(monkeypatch, fake_all_reduce)
    iterator = EpochSyncIterator([1, 2])
    assert list(iterator) == [1, 2]
    assert list(iterator) == [1, 2]


# --- Re-iterability: a factory source must survive repeated __iter__ ---


def test_factory_source_yields_a_full_pass_on_every_iteration(monkeypatch):
    """Lightning calls iter() on the train loader more than once per epoch."""
    _patch_no_distributed(monkeypatch)
    iterator = EpochSyncIterator(lambda: iter(["a", "b", "c"]))

    assert list(iterator) == ["a", "b", "c"]
    assert list(iterator) == ["a", "b", "c"]


def test_discarded_partial_pass_does_not_close_a_factory_source(monkeypatch):
    """A prefetch that is abandoned must not empty the next pass.

    `yield from` propagates close() to the sub-generator, so a shared one-shot
    source is destroyed when a partially consumed wrapper is garbage collected.
    """
    _patch_no_distributed(monkeypatch)
    iterator = EpochSyncIterator(lambda: iter(["a", "b", "c"]))

    first = iter(iterator)
    next(first)
    del first

    assert list(iterator) == ["a", "b", "c"]


def test_len_does_not_rebuild_a_factory_source_on_every_call():
    """`__len__` is called several times per epoch; rebuilding each time is waste.

    For a sequence-style factory `build_iter` constructs a DataLoader and shuffles
    the whole batch list, so this is O(corpus) per call.
    """
    calls = []

    def factory():
        calls.append(1)
        return [[0], [1], [2]]

    iterator = EpochSyncIterator(factory)

    assert len(iterator) == 3
    assert len(iterator) == 3
    assert len(iterator) == 3
    assert len(calls) == 1


def test_len_does_not_rebuild_an_unsized_factory_source_either():
    """An unsized source must be probed once, not on every `__len__` call."""
    calls = []

    def factory():
        calls.append(1)
        return iter([[0], [1]])          # a one-shot iterator has no __len__

    iterator = EpochSyncIterator(factory)

    for _ in range(3):
        with pytest.raises(TypeError):
            len(iterator)
    assert len(calls) == 1
