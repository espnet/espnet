import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Union

import torch
import torch.nn
import torch.optim
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochScheduler
from espnet2.train.reporter import Reporter


def resume(
    model: torch.nn.Module,
    optimizers: Iterable[torch.optim.Optimizer],
    epoch_schedulers: Iterable[AbsEpochScheduler],
    batch_schedulers: Iterable[AbsBatchScheduler],
    reporter: Reporter,
    output_dir: Union[str, Path],
    resume_epoch: Optional[Union[int, str]],
    resume_path: Optional[Union[str, Path]],
    map_location: str,
) -> None:
    assert check_argument_types()
    # For resuming: Specify either resume_epoch or resume_path.
    #     - resume_epoch: Load from outdir/{}epoch/.
    #     - resume_path: Load from the specified path.
    # Find the latest epoch snapshot
    if resume_epoch == "latest":
        resume_epoch = 0
        latest = None
        for p in output_dir.glob("*epoch/timestamp"):
            try:
                n = int(p.parent.name.replace("epoch", ""))
            except TypeError:
                continue
            with p.open("r") as f:
                # Read the timestamp and comparing
                date = f.read().strip()
                try:
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
            if latest is None or date > latest:
                resume_epoch = n
                latest = date

    # If not found any snapshots, then nothing is done
    if resume_epoch == 0:
        resume_epoch = None

    if resume_epoch is not None or resume_path is not None:
        if resume_path is None:
            resume_path = output_dir / f"{resume_epoch}epoch"
            logging.info(
                f"--resume_epoch {resume_epoch}: Loading from {resume_path}"
            )
        else:
            logging.info(f"--resume_path {resume_path}: Loading from {resume_path}")
        load_checkpoint(
            resume_path=resume_path,
            model=model,
            reporter=reporter,
            optimizers=optimizers,
            epoch_schedulers=epoch_schedulers,
            batch_schedulers=batch_schedulers,
            map_location=map_location,
        )


def load_checkpoint(
    resume_path: Union[str, Path],
    model: torch.nn.Module,
    reporter: Reporter,
    optimizers: Iterable[torch.optim.Optimizer],
    epoch_schedulers: Iterable[Optional[AbsEpochScheduler]],
    batch_schedulers: Iterable[Optional[AbsBatchScheduler]],
    map_location: str
):
    for key, obj in [("model", model), ("reporter", reporter)]:
        _st = torch.load(resume_path / f"{key}.pth", map_location=map_location)
        if obj is not None:
            obj.load_state_dict(_st)
    states = torch.load(resume_path / f"optim.pth")
    for o, e, b, state in zip(
        optimizers, epoch_schedulers, batch_schedulers, states
    ):
        o.load_state_dict(state["optim"])
        if e is not None:
            e.load_state_dict(state["escheduler"])
        if b is not None:
            b.load_state_dict(state["bscheduler"])


def save_checkpoint(
        save_path: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Iterable[torch.optim.Optimizer],
        epoch_schedulers: Iterable[Optional[AbsEpochScheduler]],
        batch_schedulers: Iterable[Optional[AbsBatchScheduler]],
):
    for key, obj in [("model", model), ("reporter", reporter)]:
        save_path.mkdir(parents=True, exist_ok=True)
        p = save_path / f"{key}.pth"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj.state_dict() if obj is not None else None, p)
    states = tuple(
        {
            "optim": o.state_dict(),
            "escheduler": e.state_dict() if e is not None else None,
            "bscheduler": b.state_dict() if b is not None else None,
        } for o, e, b in zip(optimizers, epoch_schedulers, batch_schedulers)
    )
    torch.save(states, save_path / f"optim.pth")

    # Write the datetime in "timestamp"
    with (save_path / "timestamp").open("w") as f:
        dt = datetime.now()
        f.write(dt.strftime("%Y-%m-%d %H:%M:%S") + "\n")
