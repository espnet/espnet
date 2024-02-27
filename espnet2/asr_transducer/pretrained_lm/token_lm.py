"""Pre-trained token-level language model definition."""

import collections
import logging
import os
from typing import Dict, List, Tuple

import torch


class PretrainedTokenLM:
    """Pre-trained token-level language model module.

    Args:
        model_path: Pre-trained model path.
        token_list: List of known tokens.
        score_weight: Weight for the outputted log-probabilities.
        device: Device to pin the state on.
        padding_id: Padding symbol ID.

    """

    def __init__(
        self,
        model_path: str,
        token_list: List[str],
        score_weight: float,
        device: str,
        padding_id: int = 0,
    ) -> None:
        """Construct a PretrainedTokenLM object."""
        super().__init__()

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Specified lm was not found at: {model_path}")

        if model_path.endswith(".pth"):
            from espnet2.asr_transducer.pretrained_lm.lstm import PretrainedTokenLSTM

            (
                model_state_dict,
                device,
                embed_size,
                hidden_size,
                num_layers,
            ) = self.get_lstm_model_parameters(model_path, device)

            self.lm = PretrainedTokenLSTM(
                len(token_list),
                embed_size,
                hidden_size,
                num_layers,
                model_state_dict,
                score_weight,
                device,
                padding_id=padding_id,
            )

            self.lm_type = "lstm"
        elif model_path.endswith(".arpa"):
            from espnet2.asr_transducer.pretrained_lm.ngram import PretrainedTokenNgram

            self.lm = PretrainedTokenNgram(model_path, token_list, score_weight, device)

            self.lm_type = "ngram"
        else:
            raise ValueError(
                "Specified LM file should be either an .pth or .arpa file. Aborting."
            )

    def get_lstm_model_parameters(
        self, model_path: str, device: str
    ) -> Tuple[Dict, torch.device, int, int, int]:
        """Get model parameters and state dict from pretrained model file.

        Args:
            model_path: Pretrained model file.
            device: Type of device to pin the model.

        Returns:
            model_state_dict: Model state dict.
            device: Type of device to pin the model.
            embed_size: Embedding size.
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.

        """
        needed_keys = ["lm.encoder.weight", "lm.rnn.weight_hh_l0", "weight_hh_l"]

        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"

        pt_state_dict = torch.load(model_path, map_location=device)
        model_state_dict = collections.OrderedDict()

        # (b-flo): dummy checking // to fix.
        if not all(nd_key in pt_state_dict.keys() for nd_key in needed_keys):
            logging.error(
                f"The p.-t. language model {model_path} isn't a valid SequentialRNNLM."
            )

        embed_size = int(pt_state_dict[needed_keys[0]].size(1))
        hidden_size = int(pt_state_dict[needed_keys[1]].size(1))
        num_layers = sum(needed_keys[2] in k for k in pt_state_dict.keys())

        for k, v in pt_state_dict.items():
            model_state_dict[k.replace("lm.", "")] = v

        return model_state_dict, device, embed_size, hidden_size, num_layers
