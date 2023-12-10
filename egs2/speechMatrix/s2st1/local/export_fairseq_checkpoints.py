# https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/enhanced_direct_s2st_discrete_units.md#pretrained-models

import logging
from typing import Any, Dict, Optional, Union

from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
import torch
from omegaconf import DictConfig, open_dict
from fairseq.checkpoint_utils import load_checkpoint_to_cpu, load_model_ensemble_and_task

logger = logging.getLogger(__name__)


def fix_checkpoint_cfg(
        filepath: str,
        arg_overrides: Optional[Dict[str, Any]] = None,
        task_arg_pops: Optional[list] = None,
):
    state = load_checkpoint_to_cpu(filepath, arg_overrides)

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
        )

    # NOTE: pop task arguments if specified
    if task_arg_pops is None:
        task_arg_pops = []

    cfg = DictConfig(cfg)
    with open_dict(cfg):
        for key in task_arg_pops:
            cfg.task.pop(key)
    state['cfg'] = cfg

    return state


def wav2vec():
    # https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/transformer_B.pt
    ckpt = 'transformer_B.pt'

    state = fix_checkpoint_cfg(
        ckpt,
        arg_overrides={"data": '.'},
        task_arg_pops=['fbank_features']  # fairseq's backward-incompatible changes
    )

    torch.save(state, 'wav2vec2_transformer_base.es.pth')


def mbart():
    # https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/unit_mBART/checkpoint.pt
    # https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/dict.txt
    ckpt = 'checkpoint.pt'
    models, _, _ = load_model_ensemble_and_task(
        [ckpt],
        arg_overrides={"data": '.'},
    )
    model = models[0]
    # print(model)
    torch.save(model.state_dict(), 'unit_mbart.pth')


def main():
    wav2vec()
    # mbart()


if __name__ == '__main__':
    main()
