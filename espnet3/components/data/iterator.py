"""Base per-epoch batch iterator for ESPnet3."""

import torch


class BaseIterator:
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
    """

    def __init__(self, iterator):
        """Wrap one rank's per-epoch batch iterator."""
        self._iterator = iterator

    def __len__(self):
        """Return the number of batches of the wrapped iterator.

        In distributed runs this is an upper bound: the epoch ends earlier on
        every rank when the first rank runs out of batches.
        """
        return len(self._iterator)

    def generate(self):
        """Yield this rank's batches."""
        yield from self._iterator

    def __iter__(self):
        """Yield batches, stopping on all ranks once any rank is exhausted."""
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
