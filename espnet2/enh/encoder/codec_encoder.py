import torch
import torchaudio.transforms as T
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer


class CodecEncoder(AbsEncoder):
    """Codec encoder for speech enhancement and separation"""

    def __init__(
        self,
        channel: int, #this is the expected output size of the continuous codec output
        codec_choice: str,
        codec_fs: int,
        sample_fs: int,
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

        self.codec_fs = codec_fs
        self.sample_fs = sample_fs
        if codec_fs != sample_fs:
            self.dac_sampler = T.Resample(sample_fs, codec_fs).to(device)
        # self.org_sampler = T.Resample(DAC_sample_rate, input_sample_rate)

        for param in self.codec.parameters():
            param.requires_grad = False
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

        if self.codec_fs != self.sample_fs:
            input = self.resample_audio(input)
        
        feature = self.codec.encode_continuous(input)

        flens = ilens.clone().apply_(lambda x: ((x//self.codec.subsample) +1 ))

        return feature, flens

    def resample_audio(self, x):
        '''
        torchaudio resample function used here only requires last dimension to be time.
        it sucks that i have to go to cpu for this. need to think how i can make this stay in gpu
        '''
        # get device
        device = x.device

        # Implement some checks on the input
        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1 #model should only be handling single channel

        # Resamples the audio from the input rate to the dac model's rate
        
        x_resamp = self.dac_sampler(x)
        
        # normalize the resampled audio, otherwise we will run into clipping issues
        x_resamp = x_resamp / torch.max(x_resamp.abs(),dim=2,keepdim=True)[0]

        return x_resamp.to(device)