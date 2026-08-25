"""Base per-epoch batch iterator for ESPnet3."""

import torch


class EpochSyncIterator:
    """Base class for the per-epoch iterators returned by DataLoaderBuilder.

    Wraps the iterator produced by an espnet2-style iter factory
    (``iter_factory.build_iter(epoch)``) and defines the iterator interface
    handed to the trainer.

    When ``torch.distributed`` is initialized, the base class synchronizes the
    end of the epoch across ranks. Iter factories like espnet2's
    ChunkIterFactory emit a data-dependent number of batches per rank,
    so under DDP each rank would otherwise end its epoch
    at a different step. Lightning then moves the exhausted rank into
    validation while the others are still training, and the two sides issue
    mismatched collectives that deadlock until the NCCL watchdog kills the
    job. Before yielding each batch, all ranks all-reduce a 1-element has-next
    flag with MIN, so the epoch ends on every rank at the same batch index.

    Without ``torch.distributed``, batches are yielded unchanged.

    Args:
        iterator (Iterable): This rank's per-epoch batch iterator, typically
            the return value of ``iter_factory.build_iter(epoch)``. It is
            iterated once per ``__iter__`` call, and ``__len__`` is delegated
            to it, so it must be sized for ``len()`` to work.

    Note:
        Subclasses should override :meth:`generate` rather than ``__iter__``,
        so that the epoch-end synchronization stays in place.

    Note:
        The synchronization costs one all-reduce of a 1-element tensor per
        batch, which is negligible next to the gradient synchronization that
        every distributed training step already performs.

    Examples:
        Without ``torch.distributed``, the wrapper is a pass-through:

        >>> list(BaseIterator([{"speech": 0}, {"speech": 1}]))
        [{'speech': 0}, {'speech': 1}]

        Under DDP, ``DataLoaderBuilder`` wraps the iter factory's iterator so
        that every rank stops at the shortest rank's batch count::

            iter_factory = ChunkIterFactory(dataset, batches=batches, ...)
            iterator = BaseIterator(iter_factory.build_iter(epoch))
            for batch in iterator:  # same number of steps on every rank
                ...
    """

    def __init__(self, iterator):
        """Wrap one rank's per-epoch batch iterator."""
        self._iterator = iterator

    def __len__(self):
        """Return the number of batches of the wrapped iterator.

        In distributed runs this is an upper bound: the epoch ends earlier on
        every rank when the first rank runs out of batches.

        Returns:
            int: The number of batches the wrapped iterator reports.

        Raises:
            TypeError: If the wrapped iterator does not implement ``__len__``.

        Examples:
            >>> len(BaseIterator([[0], [1], [2]]))
            3
        """
        return len(self._iterator)

    def generate(self):
        """Yield this rank's batches.

        This is the extension point of the class: override it to change what a
        batch looks like, or which batches are emitted. ``__iter__`` consumes
        this generator, so an override inherits the epoch-end synchronization
        for free.

        Yields:
            Any: The batches of the wrapped iterator, in order and unchanged.

        Examples:
            >>> class NonEmptyIterator(BaseIterator):
            ...     def generate(self):
            ...         for batch in super().generate():
            ...             if len(batch) > 0:
            ...                 yield batch
            >>> list(NonEmptyIterator([[0], [], [1]]))
            [[0], [1]]
        """
        yield from self._iterator

    def __iter__(self):
        """Yield batches, stopping on all ranks once any rank is exhausted.

        Without an initialized ``torch.distributed`` process group, this is
        exactly :meth:`generate`. With one, every rank all-reduces a has-next
        flag with MIN before each batch is yielded, so the ranks that still
        have batches left stop together with the first rank that does not.

        Yields:
            Any: The batches of :meth:`generate`, truncated in distributed
            runs to the batch count of the shortest rank.

        Note:
            The has-next flag lives on CUDA for the NCCL backend and on CPU
            otherwise. The device is resolved here rather than in
            ``__init__`` so that constructing the iterator never touches CUDA.

        Examples:
            >>> # Single process: all batches are yielded.
            >>> list(BaseIterator([[0], [1], [2]]))
            [[0], [1], [2]]
            >>> # Two ranks holding 3 and 2 batches: both yield 2 batches,
            >>> # so neither rank enters validation while the other trains.
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            yield from self.generate()
            return
        # NCCL only reduces CUDA tensors; gloo/mpi take CPU tensors. Resolved
        # here rather than in __init__ so construction never touches CUDA.
        if torch.distributed.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        it = self.generate()
        while True:
            try:
                batch = next(it)
                has_next = torch.ones(1, device=device)
            except StopIteration:
                batch = None
                has_next = torch.zeros(1, device=device)
            torch.distributed.all_reduce(has_next, op=torch.distributed.ReduceOp.MIN)
            # After the all-reduce, has_next is 0 on all ranks if any rank's
            # iterator is exhausted.
            if has_next.item() == 0:
                return
            yield batch
