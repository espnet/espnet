from espnet2.train.preprocessor import (
    CommonPreprocessor,
    S2TPreprocessor as espnet2S2TPreprocessor
)
from typeguard import typechecked
from typing import Dict, Union, Iterable, Collection, Optional
from pathlib import Path
import numpy as np
from transformers import PreTrainedTokenizerFast


class S2TPreprocessor(espnet2S2TPreprocessor, CommonPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: Optional[str] = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: Optional[str] = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: Optional[str] = None,
        rir_scp: Optional[str] = None,
        rir_apply_prob: float = 1.0,
        noise_scp: Optional[str] = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        short_noise_thres: float = 0.5,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
        text_prev_name: str = "text_prev",
        text_ctc_name: str = "text_ctc",
        fs: int = 16000,
        na_symbol: str = "<na>",  # text is not available e.g. for prev or ctc
        speech_length: float = 30,  # pad or trim speech to this value in seconds
        speech_resolution: float = 0.02,  # speech time resolution
        speech_init_silence: float = 1.0,  # max silence before speech for data aug
        text_prev_apply_prob: float = 0.5,  # whether to condition on text_prev
        time_apply_prob: float = 0.5,  # whether to include timestamps
        notime_symbol: str = "<notimestamps>",
        first_time_symbol: str = "<0.00>",
        last_time_symbol: str = "<30.00>",

        # This is for HF tokenizer
        tokenizer_path: Union[str, Path] = "tokenizer.json"
    ):
        super().__init__(
            train=train,
            token_type=token_type,
            token_list=token_list,
            bpemodel=bpemodel,
            text_cleaner=text_cleaner,
            g2p_type=g2p_type,
            unk_symbol=unk_symbol,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            delimiter=delimiter,
            rir_scp=rir_scp,
            rir_apply_prob=rir_apply_prob,
            noise_scp=noise_scp,
            noise_apply_prob=noise_apply_prob,
            noise_db_range=noise_db_range,
            short_noise_thres=short_noise_thres,
            speech_volume_normalize=speech_volume_normalize,
            speech_name=speech_name,
            text_name=text_name,
            fs=fs,
        )
        print(tokenizer_path)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        self.text_prev_name = text_prev_name
        self.text_ctc_name = text_ctc_name
        self.speech_length = int(speech_length * fs)
        self.speech_resolution = int(speech_resolution * fs)
        self.speech_init_silence = int(speech_init_silence * fs)
        self.text_prev_apply_prob = text_prev_apply_prob
        self.time_apply_prob = time_apply_prob

        # Obtain the token id of special tokens
        self.na_symbol = na_symbol
        self.notime = self.tokenizer.added_tokens_encoder[notime_symbol]
        self.first_time = self.tokenizer.added_tokens_encoder[first_time_symbol]
        self.last_time = self.tokenizer.added_tokens_encoder[last_time_symbol]

    @typechecked
    def _text_process(
        self, data: Dict[str, Union[str, np.ndarray]], time_shift: int
    ) -> Dict[str, np.ndarray]:

        text_names = [self.text_name, self.text_prev_name, self.text_ctc_name]
        if self.tokenizer is not None:
            for name in text_names:
                if name in data:
                    text = data[name]

                    # Remove prev text by setting it to <na>
                    if (
                        self.train
                        and name == self.text_prev_name
                        and np.random.uniform() > self.text_prev_apply_prob
                    ):
                        text = self.na_symbol

                    text = self.text_cleaner(text)

                    # Changed the following part into HF tokenizer
                    # tokens = self.tokenizer.text2tokens(text)
                    # text_ints = self.token_id_converter.tokens2ids(tokens)
                    text_ints = self.tokenizer(text)['input_ids']
                    text_ints = np.array(text_ints, dtype=np.int64)

                    # Augment text
                    if name == self.text_name:
                        # NOTE(yifan): The first token is always space
                        # which should be removed.
                        # No space is allowed between special tokens.
                        # This works for bpe, but maybe not for the other types.
                        text_ints = text_ints[1:]

                        # Remove timestamps
                        if self.train and np.random.uniform() > self.time_apply_prob:
                            # Timestamps are continuous ints
                            text_ints = text_ints[
                                np.logical_or(
                                    text_ints < self.first_time,
                                    text_ints > self.last_time,
                                )
                            ]
                            # First two tokens are <category> and <task>
                            text_ints = np.insert(text_ints, 2, self.notime)

                        # Shift timestamps
                        text_ints[
                            np.logical_and(
                                text_ints >= self.first_time,
                                text_ints <= self.last_time,
                            )
                        ] += time_shift

                    data[name] = text_ints

        return data