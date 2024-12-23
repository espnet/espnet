#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class SpeechLMInferenceOptions:
    """
    Options for inference in Speech Language Models.

    This class holds various parameters that control the behavior of the
    inference process for Speech Language Models. Users can adjust these
    parameters to fine-tune the model's output according to their needs.

    Attributes:
        device (str): The device to run the model on. Default is "cpu".
        search_algo (str): The algorithm used for searching the next token.
            Default is "sampling".
        nbest (int): The number of best candidates to consider. Default is 1.
        sampling_temperature (float): The temperature parameter for sampling.
            Higher values lead to more randomness. Default is 1.0.
        top_k (int): The number of top candidates to sample from. Default is 20.
        maxlenratio (float): The maximum length ratio for the generated output
            compared to the input. Default is 0.0.
        minlenratio (float): The minimum length ratio for the generated output
            compared to the input. Default is 0.0.
        eos (int): The end-of-sequence token ID. Default is 5.
        start (int): The start token ID. Default is 1.
        masks (torch.Tensor): Optional masks for the input sequences. Default is None.
        nq (int): Number of queries for the model. Default is None.

    Examples:
        >>> options = SpeechLMInferenceOptions(
        ...     device="cuda",
        ...     search_algo="beam_search",
        ...     nbest=5,
        ...     sampling_temperature=0.8
        ... )
        >>> print(options.device)
        "cuda"
    """

    device: str = "cpu"
    search_algo: str = "sampling"
    nbest: int = 1
    sampling_temperature: float = 1.0
    top_k: int = 20
    maxlenratio: float = 0.0
    minlenratio: float = 0.0
    eos: int = 5
    start: int = 1
    masks: torch.Tensor = None
    nq: int = None


class AbsCoreLM(torch.nn.Module, ABC):
    """
    The abstract CoreLM class for SpeechLM, which is the major component of SpeechLM.

    It supports or is going to support several styles of SpeechLM:
    Auto-Regressive (AR):
        SpearTTS: https://arxiv.org/abs/2302.03540 (TODO)
        MusicGen: https://arxiv.org/abs/2306.05284 (TODO)
        UniAudio: https://arxiv.org/abs/2310.00704

    Non-Auto-Regressive (NAR):
        SoundStorm: https://arxiv.org/abs/2305.09636 (TODO)

    Auto-Regressive + Non-Auto-Regressive (AR + NRA): Hybrid of AR and NAR.
        Vall-E: https://arxiv.org/abs/2301.02111

    For developers: to build a new core_lm model, try to follow:
        (1) Build with Espnet Espnet internal modules:
            Use modules from `espnet2.speechlm.module.transformer.py`. If you get
            some modules that are specific to your model, put them under
            `espnet2.speechlm.module.<model_name>.py`.
        (2) or, Build with HuggingFace model/modules:
            Put everything in `espnet2.speechlm.core_lm.<model_name>.py`. Usually,
            this is just a wrapper that bridges HF models into Espnet SpeechLM.
    Reminder: try to avoid any model dependency beyond espnet2.speechlm.

    Attributes:
        None

    Methods:
        forward: Abstract method for model forward pass.
        inference: Method for performing inference with the model.

    Raises:
        NotImplementedError: If the method is not implemented in a subclass.

    Examples:
        # Example subclass implementation
        class MyCoreLM(AbsCoreLM):
            def forward(self, dec_seq, dec_seq_lengths=None, enc_seq=None,
                        enc_seq_lengths=None, prefix_len=None):
                # Implementation here
                pass

            def inference(self, prefix, opts, enc_seq=None, suffix=None):
                # Implementation here
                pass
    """

    @abstractmethod
    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
            The abstract CoreLM class for SpeechLM, which is the major component of SpeechLM.

        It supports or is going to support several styles of SpeechLM:
        Auto-Regressive (AR):
              SpearTTS: https://arxiv.org/abs/2302.03540 (TODO)
              MusicGen: https://arxiv.org/abs/2306.05284 (TODO)
              UniAudio: https://arxiv.org/abs/2310.00704

        Non-Auto-Regressive (NAR):
              SoundStorm: https://arxiv.org/abs/2305.09636 (TODO)

        Auto-Regressive + Non-Auto-Regressive (AR + NRA): Hybrid of AR and NAR.
              Vall-E: https://arxiv.org/abs/2301.02111

        For developers: to build a new core_lm model, try to follow:
            (1) Build with Espnet Espnet internal modules:
                Use modules from `espnet2.speechlm.module.transformer.py`. If you get
                some modules that is specific to your model, put them under
                `espnet2.speechlm.module.<model_name>.py`.
            (2) or, Build with HuggingFace model/modules:
                Put everything in `espnet2.speechlm.core_lm.<model_name>.py`. Usually
                this is just a wrapper that bridges HF models into Espnet SpeechLM.
        Reminder: try to avoid any model dependency beyond espnet2.speechlm.
        """
        raise NotImplementedError

    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        """
            The abstract CoreLM class for SpeechLM, which is the major component of SpeechLM.

        It supports or is going to support several styles of SpeechLM:
        Auto-Regressive (AR):
              SpearTTS: https://arxiv.org/abs/2302.03540 (TODO)
              MusicGen: https://arxiv.org/abs/2306.05284 (TODO)
              UniAudio: https://arxiv.org/abs/2310.00704

        Non-Auto-Regressive (NAR):
              SoundStorm: https://arxiv.org/abs/2305.09636 (TODO)

        Auto-Regressive + Non-Auto-Regressive (AR + NRA): Hybrid of AR and NAR.
              Vall-E: https://arxiv.org/abs/2301.02111

        For developers: to build a new core_lm model, try to follow:
            (1) Build with Espnet Espnet internal modules:
                Use modules from `espnet2.speechlm.module.transformer.py`. If you get
                some modules that are specific to your model, put them under
                `espnet2.speechlm.module.<model_name>.py`.
            (2) or, Build with HuggingFace model/modules:
                Put everything in `espnet2.speechlm.core_lm.<model_name>.py`. Usually,
                this is just a wrapper that bridges HF models into Espnet SpeechLM.
        Reminder: try to avoid any model dependency beyond espnet2.speechlm.

        Attributes:
            None

        Args:
            None

        Returns:
            None

        Yields:
            None

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Examples:
            None

        Note:
            This class serves as a base for implementing specific CoreLM models.

        Todo:
            - Implement specific AR and NAR models.
            - Add documentation for each supported model.
        """
        raise NotImplementedError
