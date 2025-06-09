#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.quantization
from typeguard import typechecked

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.cls import CLSTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.utils.cli_utils import get_commandline_args


class Classification:
    """Classification class

    Examples:
        >>> import soundfile
        >>> classification_model =
            Classification("classification_config.yml", "classification_model.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> classification_model(audio)
        prediction_result (int, or list of ints)
    """

    @typechecked
    def __init__(
        self,
        classification_train_config: Union[Path, str, None] = None,
        classification_model_file: Union[Path, str, None] = None,
        device: str = "cpu",
        batch_size: int = 1,
        dtype: str = "float32",
    ):

        classification_model, classification_train_args = CLSTask.build_model_from_file(
            classification_train_config, classification_model_file, device
        )
        classification_model.to(dtype=getattr(torch, dtype)).eval()

        self.classification_model = classification_model
        self.classification_train_args = classification_train_args
        self.device = device
        self.dtype = dtype
        self.token_id_converter = TokenIDConverter(
            token_list=classification_train_args.token_list
        )

    @torch.no_grad()
    @typechecked
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[List[int], torch.Tensor, str]:
        """Inference

        Args:
            speech: Input speech data
        Returns: Tuple of
            prediction: list of ints
            scores: tensor of float, (num_classes,) corresponding to each class'
                probability
        """

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
        scores = self.classification_model.score(**batch)  # (1, num_classes)

        # scores would be a tensor of shape (batch_size, num_classes)
        # representing probabilities.
        if self.classification_model.classification_type == "multi-class":
            prediction = [torch.argmax(scores, dim=-1).item()]  # list (1,)
        elif self.classification_model.classification_type == "multi-label":
            prediction = scores > 0.5  # Fixed threshold, (1, num_labels)
            # list (num_labels,)
            prediction = torch.nonzero(prediction.squeeze(0)).squeeze(-1).tolist()
        else:
            raise NotImplementedError(
                "Unsupported classification type: "
                f"{self.classification_model.classification_type}"
            )
        prediction_string = " ".join(self.token_id_converter.ids2tokens(prediction))
        return prediction, scores.squeeze(0), prediction_string


@typechecked
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
    classification_train_config: Optional[str],
    classification_model_file: Optional[str],
    allow_variable_data_keys: bool,
    output_all_probabilities: bool,
):

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if batch_size > 1:
        # TODO(shikhar): Implement batch decoding
        raise NotImplementedError("batch decoding is not implemented for batch size >1")

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build classification
    classification_kwargs = dict(
        classification_train_config=classification_train_config,
        classification_model_file=classification_model_file,
        device=device,
        dtype=dtype,
    )
    classification = Classification(**classification_kwargs)

    # 3. Build data-iterator
    loader = CLSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=CLSTask.build_preprocess_fn(
            classification.classification_train_args, False
        ),
        collate_fn=CLSTask.build_collate_fn(
            classification.classification_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Inference
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            try:
                predictions, scores, prediction_string = classification(**batch)
                if not output_all_probabilities:
                    scores_reduced = [scores[pred].item() for pred in predictions]
                    scores = scores_reduced
                else:
                    scores = scores.tolist()

            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                predictions = [-1]
                scores = [0]
                prediction_string = ""

            # Only supporting batch_size==1
            key = keys[0]
            # Create a directory: outdir/{n}best_recog
            result_writer = writer["prediction"]
            # Write the result to each file
            result_writer["score"][key] = " ".join([str(score) for score in scores])
            result_writer["token"][key] = " ".join([str(pred) for pred in predictions])
            result_writer["text"][key] = prediction_string
            logging.info("processed {}: prediction {}".format(key, prediction_string))


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="CLS Decoding",
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
        "--classification_train_config",
        type=str,
        help="Classification training configuration",
    )
    group.add_argument(
        "--classification_model_file",
        type=str,
        help="Classification model parameter file",
    )
    group.add_argument(
        "--output_all_probabilities",
        type=str2bool,
        default=False,
        help="Output scores for all classes",
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
