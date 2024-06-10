import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer


class CodecEncoder(AbsEncoder):
    """Codec encoder for speech enhancement and separation"""

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

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        assert input.dim() == 2, "Currently only support single channel input"

        input = torch.unsqueeze(input, 1)
        
        feature = self.codec.encode_continuous(input)

        flens = ilens.clone().apply_(lambda x: ((x//self.codec.subsample) +1 ))

        return feature, flens


if __name__ == "__main__":
    device = "cuda:0"
    codec = CodecTokenizer(
        codec_choice="ESPnet",
        codec_fs=16000,
        device=device,
        dump_audio=True,
        checkpoint_path="/home/jiaqi006/code/new_espnet/soundstream_model/16khz_soundstream/train.total_count.best.pth",
        config_path="/home/jiaqi006/code/new_espnet/soundstream_model/16khz_soundstream/config.yaml",
    )

    import soundfile as sf

    waveform, sr = sf.read("447o0312_2.0302_440c0206_-2.0302_16k.wav")
    waveform = (
        torch.from_numpy(waveform).view(1, 1, -1).to(device).float()
    )  # [B, C, n_sample]
    waveform = waveform.repeat(2, 1, 1)
    
    codec_encoder = CodecEncoder(
        channel=512,
        codec_choice="ESPnet",
        codec_fs=16000,
        device=device,
        dump_audio=True,
        checkpoint_path="/home/jiaqi006/code/new_espnet/soundstream_model/16khz_soundstream/train.total_count.best.pth",
        config_path="/home/jiaqi006/code/new_espnet/soundstream_model/16khz_soundstream/config.yaml",
    )
    waveform = waveform.squeeze(1)

    ilens = torch.LongTensor([waveform.shape[-1],40000])
    frames, flens = codec_encoder(waveform, ilens)
    print(frames.shape)
    print(flens)

    waveform = waveform[:,:60000]
    ilens = torch.LongTensor([waveform.shape[-1],40000])
    frames, flens = codec_encoder(waveform, ilens)
    print(frames.shape)
    print(flens)
