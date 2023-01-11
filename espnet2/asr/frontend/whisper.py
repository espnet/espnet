import contextlib
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class WhisperFrontend(AbsFrontend):
    """Speech Representation Using Encoder Outputs from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        whisper_model: str = "small",
        freeze_weights: bool = True,
        download_dir: str = None,
    ):
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_MELS
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        assert check_argument_types()
        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters
        self.pad_or_trim = whisper.pad_or_trim

        assert whisper_model in whisper.available_models()
        self.whisper = whisper.load_model(whisper_model, download_root=download_dir)
        self.whisper.eval()

        self.freeze_weights = freeze_weights

    def output_size(self) -> int:
        return self.whisper.encoder.ln_post.normalized_shape[-1]

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
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
        whisper_encoder = self.whisper.encoder

        x = F.gelu(whisper_encoder.conv1(input))
        x = F.gelu(whisper_encoder.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = whisper_encoder.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + whisper_encoder.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            x = x[:, :max_pos, :] + whisper_encoder.positional_embedding

        for block in whisper_encoder.blocks:
            x = block(x)

        x = whisper_encoder.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - whisper_encoder.conv2.kernel_size[0]
                    + 2 * whisper_encoder.conv2.padding[0]
                )
                // whisper_encoder.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lens = self.log_mel_spectrogram(input, input_lengths)

        with torch.no_grad() if self.freeze_weights else contextlib.nullcontext():
            feats, feats_lens = self.whisper_encode(feats, feats_lens)

        return feats, feats_lens
