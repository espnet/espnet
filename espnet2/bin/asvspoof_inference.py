#!/usr/bin/env python3
import argparse
import logging
import sys
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import check_argument_types, check_return_type

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asvspoof import ASVSpoofTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.cli_utils import get_commandline_args


class SpeechAntiSpoof:
    """SpeechAntiSpoof class
    Examples:
        >>> import soundfile
        >>> speech_anti_spoof = SpeechAntiSpoof("asvspoof_config.yml", "asvspoof.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech_anti_spoof(audio)
        prediction_result (int)
    """

    def __init__(
        self,
        asvspoof_train_config: Union[Path, str] = None,
        asvspoof_model_file: Union[Path, str] = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
    ):
        assert check_argument_types()

        asvspoof_model, asvspoof_train_args = ASVSpoofTask.build_model_from_file(
            asvspoof_train_config, asvspoof_model_file, device
        )
        asvspoof_model.to(dtype=getattr(torch, dtype)).eval()

        self.asvspoof_model = asvspoof_model
        self.asvspoof_train_args = asvspoof_train_args
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> float:
        """Inference
        Args:
            data: Input speech data
        Returns:
            [prediction, scores]
        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logging.info("speech length: " + str(speech.size(1)))

        # To device
        batch = to_device(batch, device=self.device)

        # TODO1 (checkpoint 4): Forward feature extraction and encoder etc.

        if "oc_softmax_loss" in self.asvspoof_model.losses:
            pass  # TODO1 (exercise2): use loss score function to estimate score
        else:
            pass  # TODO2 (checkpoint 4): Pass the encoder result to decoder

        # TODO3 (checkpoint 4): return the prediction score
        return None


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asvspoof_train_config: Optional[str],
    asvspoof_model_file: Optional[str],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech_anti_spoof
    speech_anti_spoof_kwargs = dict(
        asvspoof_train_config=asvspoof_train_config,
        asvspoof_model_file=asvspoof_model_file,
        device=device,
        dtype=dtype,
    )
    speech_anti_spoof = SpeechAntiSpoof(
        **speech_anti_spoof_kwargs,
    )

    # 3. Build data-iterator
    loader = ASVSpoofTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASVSpoofTask.build_preprocess_fn(
            speech_anti_spoof.asvspoof_train_args, False
        ),
        collate_fn=ASVSpoofTask.build_collate_fn(
            speech_anti_spoof.asvspoof_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                score = speech_anti_spoof(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                score = 0

            # Only supporting batch_size==1
            key = keys[0]

            # Create a directory: outdir/{n}best_recog
            result_writer = writer[f"prediction"]

            # Write the result to each file
            result_writer["score"][key] = str(score)

            logging.info("processed {}: score {}".format(key, score))


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="ASVSpoof Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asvspoof_train_config",
        type=str,
        help="ASVSpoof training configuration",
    )
    group.add_argument(
        "--asvspoof_model_file",
        type=str,
        help="ASVSpoof model parameter file",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
