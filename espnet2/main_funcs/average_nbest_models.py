import logging
from pathlib import Path
from typing import Sequence

import torch
from typeguard import check_argument_types

from espnet2.train.reporter import Reporter


@torch.no_grad()
def average_nbest_models(
    output_dir: Path,
    reporter: Reporter,
    best_model_criterion: Sequence[Sequence[str]],
    nbest: int,
) -> None:
    assert check_argument_types()
    # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
    nbest_epochs = [
        (ph, k, reporter.sort_epochs_and_values(ph, k, m)[:nbest])
        for ph, k, m in best_model_criterion
        if reporter.has(ph, k)
    ]

    _loaded = {}
    for ph, cr, epoch_and_values in nbest_epochs:
        # Note that len(epoch_and_values) doesn't always equal to nbest.
        op = output_dir / f"{ph}.{cr}.ave_{len(epoch_and_values)}best.pth"
        logging.info(
            f"Averaging {len(epoch_and_values)}best models: "
            f'criterion="{ph}.{cr}": {op}'
        )

        if len(epoch_and_values) == 0:
            continue
        elif len(epoch_and_values) == 1:
            # The averaged model is same as the best model
            e, _ = epoch_and_values[0]
            op = output_dir / f"{e}epoch.pth"
            for sym_op in [
                output_dir / f"{ph}.{cr}.ave.pth",
                output_dir / f"{ph}.{cr}.ave_{len(epoch_and_values)}best.pth",
            ]:
                if sym_op.is_symlink() or sym_op.exists():
                    sym_op.unlink()
                sym_op.symlink_to(op.name)
        else:
            avg = None
            # 2.a Averaging model
            for e, _ in epoch_and_values:
                if e not in _loaded:
                    _loaded[e] = torch.load(
                        output_dir / f"{e}epoch.pth", map_location="cpu",
                    )
                states = _loaded[e]

                if avg is None:
                    avg = states
                else:
                    # Accumulated
                    for k in avg:
                        avg[k] += states[k]
            for k in avg:
                avg[k] /= len(epoch_and_values)

            # 2.b Save the ave model and create a symlink
            torch.save(avg, op)
            sym_op = output_dir / f"{ph}.{cr}.ave.pth"
            if sym_op.is_symlink() or sym_op.exists():
                sym_op.unlink()
            sym_op.symlink_to(op.name)
