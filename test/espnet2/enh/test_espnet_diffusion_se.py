import pytest
import torch

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.diffusion.score_based_diffusion import ScoreModel
from espnet2.enh.diffusion_enh import ESPnetDiffusionModel
from espnet2.enh.encoder.stft_encoder import STFTEncoder

stft_encoder = STFTEncoder(n_fft=128, hop_length=64)
stft_decoder = STFTDecoder(n_fft=128, hop_length=64)

parameters = {
    "score_model": "dcunet",
    "score_model_conf": {
        "dcunet_architecture": "DCUNet-10",
    },
    "sde": "ouve",
    "sde_conf": {},
}
diffusion_model = ScoreModel(**parameters)


@pytest.mark.parametrize(
    "encoder, decoder",
    [
        (stft_encoder, stft_decoder),
    ],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("num_spk", [1])
@pytest.mark.parametrize("normalize", [True, False])
def test_criterion_behavior_validation(
    encoder,
    decoder,
    training,
    num_spk,
    normalize,
):
    inputs = torch.randn(2, 1200)
    ilens = torch.LongTensor([1200, 1100])
    speech_refs = [torch.randn(2, 1200).float(), torch.randn(2, 1200).float()]
    if num_spk == 1:
        speech_refs = speech_refs[:1]
    enh_model = ESPnetDiffusionModel(
        encoder=encoder,
        diffusion=diffusion_model,
        decoder=decoder,
        num_spk=num_spk,
        normalize=normalize,
    )

    if training:
        enh_model.train()
    else:
        enh_model.eval()

    kwargs = {
        "speech_mix": inputs,
        "speech_mix_lengths": ilens,
        **{"speech_ref{}".format(i + 1): speech_refs[i] for i in range(num_spk)},
    }

    if training:
        loss, stats, weight = enh_model(**kwargs)
    else:
        loss, stats, weight = enh_model(**kwargs)

        enh_model.enhance(enh_model.encoder(inputs, ilens)[0])
