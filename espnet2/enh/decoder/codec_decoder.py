import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer

class CodecDecoder(AbsDecoder):
    """Codec decoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int, #this is the expected output size of the continuous codec output
        codec_choice: str,
        codec_fs: int,
        device: str = "cpu",
        dump_audio: bool = False,
        checkpoint_path: str = None,
        config_path: str = None,
        max_token_per_frame: int = 32,
    ):
        super().__init__()
        self.codec = CodecTokenizer(
            codec_choice=codec_choice,
            codec_fs=codec_fs,
            device=device,
            dump_audio=dump_audio,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        self._output_dim = channel

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch] (not used)
            fs (int): sampling rate in Hz (Not used)
        """
        # input = input.transpose(1, 2)
        # batch_size = input.shape[0]
        # wav = self.convtrans1d(input, output_size=(batch_size, 1, ilens.max()))
        print(ilens.max())
        wav = self.codec.decode_continuous(input)
        wav = wav[:,:ilens.max()]
        wav = wav.squeeze(1)

        return wav, ilens