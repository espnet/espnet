import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer

class CodecDecoder(AbsDecoder):
    """Codec decoder for speech enhancement and separation"""

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
        if self.codec_fs != self.sample_fs:
            self.dac_sampler = T.Resample(codec_fs, sample_fs).to(device)

        for param in self.codec.parameters():
            param.requires_grad = False
        self._output_dim = channel

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch] (not used)
            fs (int): sampling rate in Hz (Not used)
        """
        wav = self.codec.decode_continuous(input)
        
        #Resampling back to original sampling rate if needed
        wav = wav.unsqueeze(1)
        if self.codec_fs != self.sample_fs:
            wav = self.resample_audio(wav)
        wav = wav.squeeze(1)
        
        # T might have changed due to model. If so, fix it here
        T_origin = ilens.max()
        if wav.shape[-1] != T_origin:

            T_est = wav.shape[-1]
            if T_origin > T_est:
                wav = F.pad(wav, (0, T_origin - T_est))
            else:
                wav = wav[:, :T_origin]
        return wav, ilens

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