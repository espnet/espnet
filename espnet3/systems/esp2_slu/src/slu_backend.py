"""Two-pass SLU decoding: ``Speech2Understand`` fed a text transcript."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Union

import numpy as np
import torch

from espnet2.bin.slu_inference import Speech2Understand
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


class SLUInference:
    """``Speech2Understand`` that accepts the transcript as plain text.

    The deliberation models read a transcript alongside the speech, and the
    two stages disagree about its type. ``SLUPreprocessor`` turns it into token
    ids during training, but the ``infer`` stage has no preprocessor:
    ``InferenceRunner.forward`` hands the dataset field to the model as it
    comes, and ``SlurpDataset`` yields a string, while
    ``Speech2Understand.__call__`` expects a tensor it can index into a token
    list. This class closes that gap, tokenizing the field exactly as
    ``SLUPreprocessor._text_process`` does -- split on whitespace, then map
    through ``transcript_token_list`` with unknown words going to ``<unk>``.

    Everything else is delegated, so the decoding parameters and the returned
    ``(text, token, token_int, hypothesis)`` tuples are the ones
    ``Speech2Understand`` produces and ``src.inference.build_output`` already
    reads.

    Args:
        transcript_token_list: The word list ``train_tokenizer`` wrote, the
            same file the training config passes to the preprocessor and the
            model. Passing a different list silently shifts every id.
        unk_symbol: Symbol for words outside that list.
        **kwargs: Forwarded to :class:`espnet2.bin.slu_inference.Speech2Understand`,
            including the ``device`` the inference provider supplies.

    Examples:
        In a recipe's ``conf/inference.yaml``::

            model:
              _target_: espnet3.systems.esp2_slu.inference.SLUInference
              slu_train_config: ${exp_dir}/config.yaml
              slu_model_file: ${exp_dir}/valid.acc.ave_10best.pth
              transcript_token_list: ${data_dir}/manifest/transcript_tokens.txt
    """

    def __init__(
        self,
        transcript_token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
        **kwargs: Any,
    ) -> None:
        """Build the wrapped model and the transcript token converter."""
        self.speech2understand = Speech2Understand(**kwargs)
        self.tokenizer = build_tokenizer(token_type="word")
        self.token_id_converter = TokenIDConverter(
            token_list=transcript_token_list,
            unk_symbol=unk_symbol,
        )
        self.unk_id = self.token_id_converter.token2id[unk_symbol]

    def transcript_to_ids(self, transcript: str) -> List[int]:
        """Map one transcript to the ids the model's token list is keyed by.

        Args:
            transcript: The words on their own, with no intent label.

        Returns:
            One id per word. An empty transcript yields a single ``<unk>``
            rather than an empty sequence: a first-pass model can decode to
            nothing, and a zero-length transcript would reach BERT as an empty
            input.
        """
        tokens = self.tokenizer.text2tokens(transcript)
        if not tokens:
            return [self.unk_id]
        return self.token_id_converter.tokens2ids(tokens)

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray], transcript: str):
        """Decode one utterance.

        Args:
            speech: The waveform, as ``Speech2Understand`` takes it.
            transcript: The first-pass or reference transcript, as text.

        Returns:
            What ``Speech2Understand.__call__`` returns: a list of
            ``(text, token, token_int, hypothesis)`` tuples, best first.
        """
        transcript_ids = torch.tensor(
            self.transcript_to_ids(transcript), dtype=torch.long
        )
        return self.speech2understand(speech, transcript_ids)
