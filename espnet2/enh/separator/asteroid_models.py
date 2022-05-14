from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple
import warnings

import torch

from espnet2.enh.separator.abs_separator import AbsSeparator


class AsteroidModel_Converter(AbsSeparator):
    def __init__(
        self,
        encoder_output_dim: int,
        model_name: str,
        num_spk: int,
        pretrained_path: str = "",
        loss_type: str = "si_snr",
        **model_related_kwargs,
    ):
        """The class to convert the models from asteroid to AbsSeprator.

        Args:
            encoder_output_dim: input feature dimension, default=1 after the NullEncoder
            num_spk: number of speakers
            loss_type: loss type of enhancement
            model_name: Asteroid model names, e.g. ConvTasNet, DPTNet. Refers to
                        https://github.com/asteroid-team/asteroid/
                        blob/master/asteroid/models/__init__.py
            pretrained_path: the name of pretrained model from Asteroid in HF hub.
                             Refers to: https://github.com/asteroid-team/asteroid/
                             blob/master/docs/source/readmes/pretrained_models.md and
                             https://huggingface.co/models?filter=asteroid
            model_related_kwargs: more args towards each specific asteroid model.
        """
        super(AsteroidModel_Converter, self).__init__()

        assert (
            encoder_output_dim == 1
        ), encoder_output_dim  # The input should in raw-wave domain.

        # Please make sure the installation of Asteroid.
        # https://github.com/asteroid-team/asteroid
        from asteroid import models

        model_related_kwargs = {
            k: None if v == "None" else v for k, v in model_related_kwargs.items()
        }
        # print('args:',model_related_kwargs)

        if pretrained_path:
            model = getattr(models, model_name).from_pretrained(pretrained_path)
            print("model_kwargs:", model_related_kwargs)
            if model_related_kwargs:
                warnings.warn(
                    "Pratrained model should get no args with %s" % model_related_kwargs
                )

        else:
            model_name = getattr(models, model_name)
            model = model_name(**model_related_kwargs)

        self.model = model
        self._num_spk = num_spk

        self.loss_type = loss_type
        if loss_type != "si_snr":
            raise ValueError("Unsupported loss type: %s" % loss_type)

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
        additional: Optional[Dict] = None,
    ):
        """Whole forward of asteroid models.

        Args:
            input (torch.Tensor): Raw Waveforms [B, T]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data included in model

        Returns:
            estimated Waveforms(List[Union(torch.Tensor]): [(B, T), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, T),
                'mask_spk2': torch.Tensor(Batch, T),
                ...
                'mask_spkn': torch.Tensor(Batch, T),
            ]
        """

        if hasattr(self.model, "forward_wav"):
            est_source = self.model.forward_wav(input)  # B,nspk,T or nspk,T
        else:
            est_source = self.model(input)  # B,nspk,T or nspk,T

        if input.dim() == 1:
            assert est_source.size(0) == self.num_spk, est_source.size(0)
        else:
            assert est_source.size(1) == self.num_spk, est_source.size(1)

        est_source = [es for es in est_source.transpose(0, 1)]  # List(M,T)
        masks = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(self.num_spk)], est_source)
        )
        return est_source, ilens, masks

    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output with waveforms."""
        return self.forward(input, ilens)

    @property
    def num_spk(self):
        return self._num_spk


if __name__ == "__main__":
    mixture = torch.randn(3, 16000)
    print("mixture shape", mixture.shape)

    net = AsteroidModel_Converter(
        model_name="ConvTasNet",
        encoder_output_dim=1,
        num_spk=2,
        loss_type="si_snr",
        pretrained_path="mpariente/ConvTasNet_WHAM!_sepclean",
    )
    print("model", net)
    output, *__ = net(mixture)
    output, *__ = net.forward_rawwav(mixture, 111)
    print("output spk1 shape", output[0].shape)

    net = AsteroidModel_Converter(
        encoder_output_dim=1,
        num_spk=2,
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
    print("Finished", output[0].shape)
