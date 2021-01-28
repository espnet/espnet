from collections import OrderedDict
from typing import List
from typing import Tuple

import torch

from espnet2.enh.separator.abs_separator import AbsSeparator


class AsteroidModel_Converter(AbsSeparator):
    """The class to convert the models from asteroid to AbsEnhancement net."""

    def __init__(
        self,
        encoder_output_dim: int,
        model_name: str,
        num_spk: int,
        pretrained_path: str = "",
        loss_type: str = "si_snr",
        **model_related_kwargs,
    ):
        super(AsteroidModel_Converter, self).__init__()
        assert encoder_output_dim == 1, encoder_output_dim # The input should in raw-wave domain.

        # Please make sure the installation of Asteroid.
        # https://github.com/asteroid-team/asteroid
        from asteroid import models

        model_related_kwargs = {
            k: None if v == "None" else v for k, v in model_related_kwargs.items()
        }
        # print('args:',model_related_kwargs)

        if pretrained_path:
            model = eval(
                "models.{}.from_pretrained('{}')".format(model_name, pretrained_path)
            )
        else:
            model_name = getattr(models, model_name)
            model = model_name(n_src=num_spk, **model_related_kwargs)

        self.model = model
        self._num_spk = num_spk

        self.loss_type = loss_type
        if loss_type != "si_snr":
            raise ValueError("Unsupported loss type: %s" % loss_type)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None):

        est_source = self.model(input)  # B,nspk,T or nspk,T
        if input.dim() == 1:
            assert est_source.size(0) == self.num_spk, est_source.size(0)
        else:
            assert est_source.size(1) == self.num_spk, est_source.size(1)

        est_source = [es for es in est_source.transpose(0, 1)]  # List(M,T)
        masks = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(self.num_spk)], est_source)
        )
        return est_source, ilens, masks

    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output with waveforms.
        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            predcited speech [Batch, num_speaker, sample]
            output lengths
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, sample),
                'spk2': torch.Tensor(Batch, sample),
                ...
                'spkn': torch.Tensor(Batch, sample),
            ]
        """
        return self.forward(input, ilens)

    @property
    def num_spk(self):
        return self._num_spk

    def process_targets(
        self, input: torch.Tensor, target: List[torch.Tensor], ilens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return target, ilens


if __name__ == "__main__":
    mixture = torch.randn(3, 16000)
    print("mixture shape", mixture.shape)
    # net = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM!_sepclean')
    # print("model", net)
    # output = net(mixture)
    # print("output shape",output.shape)

    net = AsteroidModel_Converter(
        model_name="ConvTasNet",
        n_src=2,
        loss_type="si_snr",
        pretrained_path="mpariente/ConvTasNet_WHAM!_sepclean",
    )
    # print("model", net)
    # output, *__ = net(mixture)
    output, *__ = net.forward_rawwav(mixture, 111)
    print("output spk1 shape", output[0].shape)

    net = AsteroidModel_Converter(
        model_name="ConvTasNet",
        n_src=2,
        loss_type="si_snr",
        out_chan=None,
        n_blocks=2,
        n_repeats=2,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="sigmoid",
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
    )
    print("\n\nmodel", net)
    output, *__ = net(mixture)
    print("output spk1 shape", output[0].shape)