import copy
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug

N_SAMPLES = 480000 # for input wav padding

class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        input_size: int = 1,
        fs: Union[int, str] = 16000,
        n_fft: int = 400,
        win_length: int = None,
        hop_length: int = 160,
        n_mels: int = 80,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
    ):
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && ./installers/install_whisper.sh"
            )
            raise e

        assert check_argument_types()
        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            raise ValueError(
                "whisper is trained on 16kHz audios, set fs=16000 instead of {}".format(
                    fs
                )
            )

        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        if n_fft != 400 or self.win_length != 400 or hop_length != 160 or n_mels != 80:
            raise ValueError(
                "Please use STFT settings under which whisper is trained:\n"
                + "  n_fft = 400, win_length = 400, hop_length = 160, n_mels = 80\n"
                + "  you set n_fft = {}, win_length = {}, hop_length = {}, n_mels = {}".format(
                    self.n_fft, self.win_length, self.hop_length, self.n_mels
                )
            )

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(whisper_model, download_root=download_dir)
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim

    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor, 
        length: int = N_SAMPLES, 
        axis: int = -1,
    ) -> torch.Tensor:
        """
        Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                        dim=axis, 
                        index=torch.arange(length).to(array.device)
                    )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array


    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)

        for l, block in enumerate(self.encoders.blocks):
            x = block(x)
            if l < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        x = self.encoders.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats, feats_lens = self.specaug(feats, feats_lens)

        xs_pad, olens = self.whisper_encode(feats, feats_lens)

        return xs_pad, olens, None
