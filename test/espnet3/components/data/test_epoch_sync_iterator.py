import pytest

import espnet3.components.data.epoch_sync_iterator as epoch_sync_iterator_module
from espnet3.components.data.epoch_sync_iterator import EpochSyncIterator


def _patch_no_distributed(monkeypatch):
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed, "is_initialized", lambda: False
    )


def _patch_dist_gloo(monkeypatch, all_reduce):
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed, "get_backend", lambda: "gloo"
    )
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed, "all_reduce", all_reduce
    )


# --- Non-distributed passthrough ---


def test_yields_all_batches_without_distributed(monkeypatch):
    _patch_no_distributed(monkeypatch)

    def unexpected_all_reduce(tensor, op=None):
        raise AssertionError("all_reduce must not be called without distributed")

    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.distributed,
        "all_reduce",
        unexpected_all_reduce,
    )
    iterator = EpochSyncIterator(["a", "b", "c"])
    assert list(iterator) == ["a", "b", "c"]


def test_len_delegates_to_the_source():
    assert len(EpochSyncIterator([1, 2, 3])) == 3


def test_len_raises_when_the_source_is_unsized():
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
        return iter([[0], [1]])  # a one-shot iterator has no __len__

    iterator = EpochSyncIterator(factory)

    for _ in range(3):
        with pytest.raises(TypeError):
            len(iterator)
    assert len(calls) == 1


def test_len_does_not_mask_a_factory_that_raises_type_error():
    """A TypeError raised inside the factory itself must propagate.

    Only the len() probe of the built pass may be interpreted as "unsized";
    swallowing factory bugs would misreport a broken source as an unsized one.
    """

    def broken_factory():
        raise TypeError("real bug inside the factory")

    with pytest.raises(TypeError, match="real bug inside the factory"):
        len(EpochSyncIterator(broken_factory))


def test_len_probe_pass_is_handed_to_the_first_iteration():
    """The pass built by the length probe must feed the first `__iter__`.

    Without reuse, one epoch costs three factory calls on the sequence path
    (length probe, Lightning's setup pass, the epoch-loop re-iter), each
    O(corpus).
    """
    calls = []

    def factory():
        calls.append(1)
        return [[0], [1]]

    iterator = EpochSyncIterator(factory)

    assert len(iterator) == 2
    assert list(iterator) == [[0], [1]]
    assert len(calls) == 1

    assert list(iterator) == [[0], [1]]
    assert len(calls) == 2


def test_unsized_probe_pass_is_reused_too():
    """A generator built for the probe is unstarted, so the first pass can use it."""
    calls = []

    def factory():
        calls.append(1)
        return iter([[0], [1]])

    iterator = EpochSyncIterator(factory)

    with pytest.raises(TypeError):
        len(iterator)
    assert list(iterator) == [[0], [1]]
    assert len(calls) == 1


def test_abandoned_probe_pass_does_not_starve_the_next_pass():
    """A handed-out probe pass that is dropped mid-way must not be reused."""

    def factory():
        return iter([[0], [1], [2]])

    iterator = EpochSyncIterator(factory)

    with pytest.raises(TypeError):
        len(iterator)
    prefetch = iter(iterator)
    assert next(prefetch) == [0]
    del prefetch  # Lightning discards its setup pass like this.

    assert list(iterator) == [[0], [1], [2]]


def test_nccl_backend_resolves_a_cuda_device(monkeypatch):
    """The NCCL branch selects a CUDA device for the has-next flag.

    NCCL only reduces CUDA tensors, so the sync tensor must live on the
    current CUDA device; run the branch with the CUDA calls stubbed so the
    test needs no GPU.
    """
    dist = epoch_sync_iterator_module.torch.distributed
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch.cuda, "current_device", lambda: 0
    )
    seen_devices = []
    real_ones = epoch_sync_iterator_module.torch.ones
    real_zeros = epoch_sync_iterator_module.torch.zeros
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch,
        "ones",
        lambda n, device=None: (seen_devices.append(device), real_ones(n))[1],
    )
    monkeypatch.setattr(
        epoch_sync_iterator_module.torch,
        "zeros",
        lambda n, device=None: (seen_devices.append(device), real_zeros(n))[1],
    )

    assert list(EpochSyncIterator([[0], [1]])) == [[0], [1]]
    assert seen_devices, "the sync branch never created a has-next flag"
    assert all(d.type == "cuda" for d in seen_devices)
