import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.forward_adaptor import ForwardAdaptor
from espnet2.train.abs_e2e import AbsE2E
from espnet2.utils.fileio import DatadirWriter


@torch.no_grad()
def collect_stats(
    model: AbsE2E,
    train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
    output_dir: Union[str, Path],
    ngpu: Optional[int],
    log_interval: Optional[int],
) -> None:
    """Perform on collect_stats mode.

    Running for deriving the shape information from data
    and gathering statistics.
    This method is used before executing train().

    """
    assert check_argument_types()
    output_dir = Path(output_dir)

    for itr, mode in zip([train_iter, eval_iter], ["train", "eval"]):
        if log_interval is None:
            log_interval = max(len(itr) // 20, 10)

        sum_dict = defaultdict(lambda: 0)
        sq_dict = defaultdict(lambda: 0)
        count_dict = defaultdict(lambda: 0)

        with DatadirWriter(output_dir / mode) as writer:
            for iiter, (keys, batch) in enumerate(itr, 1):
                batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

                # 1. Write shape file
                for name in batch:
                    if name.endswith("_lengths"):
                        continue
                    for i, (k, data) in enumerate(zip(keys, batch[name])):
                        if f"{name}_lengths" in batch:
                            lg = int(batch[f"{name}_lengths"][i])
                            shape = ",".join(map(str, (lg,) + data.shape[1:]))
                        else:
                            shape = ",".join(map(str, data.shape))
                        writer[f"{name}_shape"][k] = shape

                # 2. Extract feats and calc sum and square sum
                if ngpu <= 1:
                    data = model.collect_feats(**batch)
                else:
                    # Note that data_parallel can parallelize only "forward()"
                    data = data_parallel(
                        ForwardAdaptor(model, "collect_feats"),
                        (),
                        range(ngpu),
                        module_kwargs=batch,
                    )
                for k, v in data.items():
                    if k.endswith("_lengths"):
                        continue
                    if f"{k}_lengths" in data:
                        # value: (Batch, Length, Dim, ...)
                        # -> Summation over batchxlength
                        ind = (0, 1)
                        count = v.size(0) * v.size(1)
                    else:
                        # value: (Batch, Dim, ...)
                        # -> Summation over batch
                        ind = 0
                        count = v.size(0)
                    v = v.cpu()
                    v.masked_fill_(make_pad_mask(data[f"{k}_lengths"], v, 1), 0.0)
                    sum_dict[k] += v.sum(ind).cpu().numpy()
                    sq_dict[k] += (v ** 2).sum(ind).cpu().numpy()
                    count_dict[k] += count

                if iiter % log_interval == 0:
                    logging.info(f"Niter: {iiter}")

        for key in sum_dict:
            np.savez(
                output_dir / mode / f"{key}_stats.npz",
                count=count_dict[key],
                sum=sum_dict[key],
                sum_square=sq_dict[key],
                )
        with (output_dir / mode / "shape_keys").open("w") as f:
            f.write(
                "\n".join(filter(lambda x: not x.endswith("_lengths"), batch))
                + "\n"
            )
        with (output_dir / mode / "stats_keys").open("w") as f:
            f.write("\n".join(sum_dict) + "\n")