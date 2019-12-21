import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union

import torch
import torch.nn
import torch.optim
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsScheduler
from espnet2.train.reporter import Reporter


def resume(
    model: torch.nn.Module,
    optimizers: Iterable[torch.optim.Optimizer],
    schedulers: Iterable[AbsScheduler],
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
            logging.info(f"--resume_epoch {resume_epoch}: Loading from {resume_path}")
        else:
            logging.info(f"--resume_path {resume_path}: Loading from {resume_path}")
        load_checkpoint(
            resume_path=resume_path,
            model=model,
            reporter=reporter,
            optimizers=optimizers,
            schedulers=schedulers,
            map_location=map_location,
        )


def load_checkpoint(
    resume_path: Union[str, Path],
    model: torch.nn.Module,
    reporter: Reporter,
    optimizers: Iterable[torch.optim.Optimizer],
    schedulers: Iterable[Optional[AbsScheduler]],
    map_location: str,
):
    for key, obj in [("model", model), ("reporter", reporter)]:
        _st = torch.load(resume_path / f"{key}.pth", map_location=map_location)
        if obj is not None:
            obj.load_state_dict(_st)
    states = torch.load(resume_path / f"optim.pth")
    for o, s, state in zip(optimizers, schedulers, states):
        o.load_state_dict(state["optim"])
        if s is not None:
            s.load_state_dict(state["scheduler"])


def save_checkpoint(
    save_path: Union[str, Path],
    model: torch.nn.Module,
    reporter: Reporter,
    optimizers: Iterable[torch.optim.Optimizer],
    schedulers: Iterable[Optional[AbsScheduler]],
):
    for key, obj in [("model", model), ("reporter", reporter)]:
        save_path.mkdir(parents=True, exist_ok=True)
        p = save_path / f"{key}.pth"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj.state_dict() if obj is not None else None, p)
    states = tuple(
        {
            "optim": o.state_dict(),
            "scheduler": s.state_dict() if s is not None else None,
        }
        for o, s in zip(optimizers, schedulers)
    )
    torch.save(states, save_path / f"optim.pth")

    # Write the datetime in "timestamp"
    with (save_path / "timestamp").open("w") as f:
        dt = datetime.now()
        f.write(dt.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def load_pretrained_model(
    pretrain_path: Union[str, Path],
    model: torch.nn.Module,
    pretrain_key: str = None,
    map_location: str = "cpu",
):
    """Load pre-trained model

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/encoder.pth", model, "encoder")
    """
    if pretrain_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, pretrain_key)

    state_dict = obj.state_dict()
    pretrained_dict = torch.load(pretrain_path, map_location=map_location)
    # Ignores the parameters not existing in the train-model
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in state_dict
    }
    state_dict.update(pretrained_dict)
    obj.load_state_dict(state_dict)
