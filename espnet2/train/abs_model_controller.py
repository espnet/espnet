from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List, Union, Mapping, Sequence

import torch
import numpy as np

from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention


class AbsModelController(torch.nn.Module, ABC):
    """The common abstract class among each tasks

    FIXME(kamo): Is Controller a good name?

    "Controller" is referred to as a class which inherits torch.nn.Module,
    and makes the dnn-models forward as its member field,
    a.k.a delegate pattern,
    and defines "loss", "stats", and "weight" for the task.

    If you intend to implement new task in ESPNet,
    the model must inherit this class.
    In other words, the "mediator" objects between
    our training system and the your task class are
    just only these three values.

    Example:
        >>> from espnet2.tasks.abs_task import AbsTask
        >>> class YourTask(AbsTask):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight
        >>> class YourController(AbsModelController):
        ...     def forward(self, input, input_lengths):
        ...         ...
        ...         return loss, stats, weight

    """

    @abstractmethod
    def forward(self, **batch: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError
        return loss, stats, weight

    def plot_and_save_attentions(self,
                                 output_dir: Union[Path, str],
                                 ids: Sequence[str],
                                 batch: Mapping[str, torch.Tensor]):
        assert len(next(iter(batch.values()))) == len(ids), \
            (len(next(iter(batch.values()))), len(ids))

        att_dict = self.calculate_all_attentions(**batch)
        output_dir = Path(output_dir)

        for k, att_list in att_dict.items():
            assert len(att_list) == len(ids), (len(att_list), len(ids))

            for id_, att in zip(ids, att_list):
                fig = self.plot_attention(att)
                p = (output_dir / id_ / (k + '.png'))
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p)

    @staticmethod
    def plot_attention(att_w: Union[torch.Tensor, np.ndarray]):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        if isinstance(att_w, torch.Tensor):
            att_w = att_w.detach().cpu().numpy()

        if att_w.ndim == 2:
            att_w = att_w[None]
        elif att_w.ndim > 3 or att_w.ndim == 1:
            raise RuntimeError(f'Must be 2 or 3 dimension: {att_w.ndim}')

        w, h = plt.figaspect(1.0 / len(att_w))
        fig = plt.Figure(figsize=(w * 2, h * 2))
        axes = fig.subplots(1, len(att_w))

        if len(att_w) == 1:
            axes = [axes]

        for ax, aw in zip(axes, att_w):
            ax.imshow(aw.astype(np.float32), aspect='auto')
            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        return fig

    @torch.no_grad()
    def calculate_all_attentions(self, **batch: torch.Tensor) \
            -> Dict[str, List[torch.Tensor]]:
        """Derive the outputs from the all attention layers

        Args:
            batch: same as forward
        Returns:
            return_dict: A dict of a list of tensor.
            key_names x batch x (D1, D2, ...)

        """
        # TODO(kamo): Implement the base class representing Attention
        #  Now only MultiHeadedAttention is supported

        # 1. Derive the key names e.g. input, output
        keys = []
        for k in batch:
            if k + '_lengths' in batch:
                keys.append(k)
        bs = len(batch[keys[0]])
        assert all(len(v) == bs for v in batch.values()), \
            {k: v.shape for k, v in batch.items()}

        # 2. Register forward_hook fn to save the output from specific layers
        outputs = {}
        handles = {}
        for name, modu in self.named_modules():
            def hook(module, input, output, name=name):
                outputs[name] = output

            if isinstance(modu, MultiHeadedAttention):
                handle = modu.register_forward_hook(hook)
                handles[name] = handle

        # 3. Just forward one by one sample.
        # Batch-mode can't be used to keep small requirements for each models.
        return_dict = defaultdict(list)
        for ibatch in range(bs):
            # *: (B, L, ...) -> (1, L2, ...)
            _sample = \
                {k: batch[k][ibatch, None, :batch[k + '_lengths'][ibatch]]
                 for k in keys}

            # *_lengths: (B,) -> (1,)
            _sample.update(
                {k + '_lengths': batch[k + '_lengths'][ibatch, None]
                 for k in keys})

            self(**_sample)

            # Derive the attention results
            for name, output in outputs.items():
                return_dict[name].append(output)

        # 4. Remove all hooks
        for _, handle in handles.items():
            handle.remove()

        return return_dict
