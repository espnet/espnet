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


def _to_batched_tensor(data: np.ndarray, ndim: int) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    while data.dim() < ndim:
        data = data.unsqueeze(0)
    return data


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
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.token_id_converter = TokenIDConverter(
            token_list=classification_train_args.token_list
        )

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
        text: Optional[Union[torch.Tensor, np.ndarray]] = None,
        text_lengths: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[List[List[int]], torch.Tensor, List[str]]:
        """Inference

        Args:
            speech: Input speech data
            speech_lengths: Length of input speech data
            text: Input text data
            text_lengths: Length of input text data
        Returns: Tuple of
            prediction: list of list of ints
            scores: tensor of float, (batch_size, num_classes) corresponding to each class'
                probability
        """

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = _to_batched_tensor(speech, ndim=2)
            speech = speech.to(getattr(torch, self.dtype))
        if isinstance(speech_lengths, np.ndarray):
            speech_lengths = _to_batched_tensor(speech_lengths, ndim=1)
            speech_lengths = speech_lengths.to(torch.long)

        batch = {"speech": speech, "speech_lengths": speech_lengths}
        logging.info(f"speech length: {speech_lengths.tolist()}")

        # process text
        if text is not None:
            assert (
                self.classification_model.text_encoder is not None
            ), "Text input provided but model has no text encoder."
            if isinstance(text, np.ndarray):
                text = _to_batched_tensor(text, ndim=2)
                text = text.to(torch.long)
            if isinstance(text_lengths, np.ndarray):
                text_lengths = _to_batched_tensor(text_lengths, ndim=1)
                text_lengths = text_lengths.to(torch.long)

            batch["text"] = text
            batch["text_lengths"] = text_lengths
            logging.info(f"text_lengths: {text_lengths.tolist()}")

        # To device
        batch = to_device(batch, device=self.device)
        scores = self.classification_model.score(**batch)  # (batch_size, num_classes)

        # scores would be a tensor of shape (batch_size, num_classes)
        # representing probabilities.
        if self.classification_model.classification_type == "multi-class":
            # list of list of single elem
            prediction = torch.argmax(scores, dim=-1).unsqueeze(-1).tolist()
        elif self.classification_model.classification_type == "multi-label":
            prediction = scores > 0.5  # Fixed threshold, (batch_size, num_labels)
            # list (batch_size, num_labels) som of which maybe empty
            prediction = [np.nonzero(row)[0].tolist() for row in prediction]
        else:
            raise NotImplementedError(
                "Unsupported classification type: "
                f"{self.classification_model.classification_type}"
            )
        prediction_strings = [
            " ".join(self.token_id_converter.ids2tokens(pred)) for pred in prediction
        ]
        return prediction, scores, prediction_strings


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
    # if batch_size > 1:
    #     # TODO(shikhar): Implement batch decoding
    #     raise NotImplementedError("batch decoding is not implemented for batch size >1")

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
            # print(batch)
            # batch = {k: v[0] for k, v in batch.items()}

            try:
                predictions, scores, prediction_string = classification(**batch)
                if output_all_probabilities:
                    scores = scores.tolist()
                else:
                    scores_reduced = [
                        [scores[p].item() for p in pred] for pred in predictions
                    ]
                    scores = scores_reduced

            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                predictions = [[-1] * batch_size]
                scores = [[0] * batch_size]
                prediction_string = [[""] * batch_size]

            result_writer = writer["prediction"]
            for key, score, prediction, prediction_str_i in zip(
                keys, scores, predictions, prediction_string
            ):
                result_writer["score"][key] = " ".join([str(sc) for sc in score])
                result_writer["token"][key] = " ".join(
                    [str(pred) for pred in prediction]
                )
                result_writer["text"][key] = prediction_str_i
                logging.info(
                    "processed {}: prediction {}".format(key, prediction_str_i)
                )


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
