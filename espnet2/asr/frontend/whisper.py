import copy
import logging
from re import I
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend

# need to pad/trim waveform to this length (30 sec),
# whisper would complain otherwise
WHISPER_WAV_INPUT_LEN = 480000

class WhisperFrontend(AbsFrontend):
    """Speech Representation Using Encoder Outputs from OpenAI's Whisper Model:
       URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 400,
        win_length: int = None,
        hop_length: int = 160,
        n_mels: int = 80,
        whisper_model: str = 'small',
        download_dir: str = None,
    ):
        try:
            import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print("Please install whisper with: cd ${MAIN_ROOT}/tools && ./installers/install_whisper.sh")
            raise e

        assert check_argument_types()
        super().__init__()

        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        if fs != 16000:
            raise ValueError('whisper is trained on 16kHz audios, set fs=16000 instead of {}'.format(fs))

        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        if n_fft != 400 or self.win_length != 400 or hop_length != 160 or n_mels != 80:
            raise ValueError('Please use STFT settings under which whisper is trained:\n' + \
                             '  n_fft = 400, win_length = 400, hop_length = 160, n_mels = 80\n' + \
                             '  you set n_fft = {}, win_length = {}, hop_length = {}, n_mels = {}'.format(
                                self.n_fft, self.win_length, self.hop_length, self.n_mels
                            ))
        
        self.mel_filters = whisper.audio.mel_filters
        self.pad_or_trim = whisper.pad_or_trim

        assert whisper_model in whisper.available_models()
        self.whisper = whisper.load_model(
                                    whisper_model,
                                    download_root=download_dir
                                )
        self.whisper.eval()

    def output_size(self) -> int:
        return self.whisper.encoder.ln_post.normalized_shape[-1]

    def calc_whisper_encode_olens(
        self, 
        mel_lens: torch.Tensor
    ) -> torch.Tensor:
        return 1 + ( mel_lens - \
                    self.whisper.encoder.conv2.kernel_size[0] + \
                    2 * self.whisper.encoder.conv2.padding[0] ) // \
                    self.whisper.encoder.conv2.stride[0]

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
                    audio, 
                    self.n_fft, 
                    self.hop_length, 
                    window=window,
                    return_complex=True
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
        
        log_spec = torch.maximum(log_spec, log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor, 
    ) -> torch.Tensor:
        whisper_encoder = self.whisper.encoder
        
        x = F.gelu(whisper_encoder.conv1(input))
        x = F.gelu(whisper_encoder.conv2(x))
        x = x.permute(0, 2, 1)

        x = (x + whisper_encoder.positional_embedding[:x.size(1), :]).to(x.dtype)

        for block in whisper_encoder.blocks:
            x = block(x)

        x = whisper_encoder.ln_post(x)

        return x

    def forward(
        self, 
        input: torch.Tensor, 
        input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lens = self.log_mel_spectrogram(input, input_lengths)

        feats = self.whisper_encode(feats)
        feats_lens = self.calc_whisper_encode_olens(feats_lens)

        return feats, feats_lens