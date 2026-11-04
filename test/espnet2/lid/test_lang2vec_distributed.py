"""Distributed training with missing per-language auxiliary targets."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _train_with_missing_targets(rank, rendezvous, args):
    from espnet2.tasks.lid import LIDTask
    from espnet2.torch_utils.recursive_op import recursive_average

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.manual_seed(10)
        model = LIDTask.build_model(args)
        distributed = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
        distributed.train()
        optimizer = torch.optim.SGD(distributed.parameters(), lr=0.001)
        for step in range(2):
            # Alternate the observed rank to exercise changing unused heads.
            observed = rank == step
            torch.manual_seed(20 + rank + step)
            targets = (
                torch.randn(2, 3) if observed else torch.full((2, 3), float("nan"))
            )
            optimizer.zero_grad()
            loss, stats, weight = distributed(
                torch.randn(2, 1600),
                torch.tensor([1600, 980]),
                lid_labels=torch.tensor([0, 1]),
                lang2vecs=targets,
                lang2vec_mask=torch.full((2, 1), int(observed)),
            )
            assert torch.isfinite(loss)
            local = {name: value.item() for name, value in stats.items()}
            gathered = [None, None]
            dist.all_gather_object(gathered, local)

            # This is the trainer's actual reduction, before backward. Different
            # keys on a masked rank used to cause mismatched collectives here.
            averaged, total_weight = recursive_average(
                {name: value.reshape(1) for name, value in stats.items()},
                weight.reshape(1),
                distributed=True,
            )
            assert total_weight.item() == 4
            assert list(gathered[0]) == list(gathered[1])
            for name, value in averaged.items():
                expected = torch.tensor([item[name] for item in gathered]).nanmean()
                torch.testing.assert_close(value, expected)
                if "lang2vec_loss" in name:
                    # Missing targets must not introduce a zero into the mean.
                    torch.testing.assert_close(
                        value, torch.tensor(gathered[step][name])
                    )

            before = model.loss.weight.detach().clone()
            loss.backward()
            assert torch.isfinite(model.loss.weight.grad).all()
            optimizer.step()
            assert torch.isfinite(model.loss.weight).all()
            assert not torch.equal(model.loss.weight, before)
    finally:
        dist.destroy_process_group()


@pytest.mark.execution_timeout(120)
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_distributed_training_with_missing_targets(tiny_geolid, tmp_path):
    args, _, _ = tiny_geolid
    rendezvous = (tmp_path / "distributed_init").resolve().as_uri()
    context = mp.spawn(
        _train_with_missing_targets, args=(rendezvous, args), nprocs=2, join=False
    )
    try:
        while not context.join():
            pass
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
