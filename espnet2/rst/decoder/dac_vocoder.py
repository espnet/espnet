"""DAC vocoder: DAC decoder from 50 Hz SSL features to 48 kHz waveform.

Sidon (Nakata et al., arXiv:2509.17052) synthesises 48 kHz speech from
w2v-BERT 2.0 layer-8 features with the decoder half of the Descript Audio
Codec (Kumar et al., NeurIPS 2023): a stack of Snake-activated residual units
and transposed convolutions with strides [8, 5, 4, 3, 2], i.e. a 960x
upsampling of one 20 ms frame to 960 samples at 48 kHz.

The decoder is reproduced here rather than imported so the recipe does not
depend on ``descript-audio-codec`` (which pulls in ``audiotools`` and a
sizeable tree of its own dependencies for a ~90-line module). The module
tree, parameter names and initialisation follow ``dac.model.dac.Decoder``
exactly, so a checkpoint saved by that class loads into this one without
renaming, and the published Sidon vocoder, distributed only as a frozen
TorchScript graph, can be recovered into it with
``load_official_torchscript`` below. Snake, ResidualUnit and DecoderBlock
are Copyright (c) 2023-present Descript, MIT licence.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm


def _wn_conv1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _wn_conv_transpose1d(*args, **kwargs) -> nn.Module:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# The three blocks below are the decoder of the official Sidon release (DAC
# geometry). They are kept as their own modules rather than built from
# espnet2.gan_codec because local/convert_official_sidon_vocoder.py loads the
# published TorchScript weights into them bit-exactly: gan_codec's Snake1d
# shares one alpha across channels (the release has one per channel) and its
# DAC codec decodes with SEANet blocks, so neither can take those weights.
# The HiFi-GAN alternative (HiFiGANVocoder) wraps espnet2.gan_tts.
class Snake1d(nn.Module):
    """Periodic Snake activation, x + sin^2(alpha x) / alpha, per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            _wn_conv_transpose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DACVocoder(nn.Module):
    """DAC decoder generator.

    Args:
        input_dim: SSL feature dimension (1024 for w2v-BERT 2.0).
        channels: width after the input convolution; halves at every block.
            The published Sidon vocoder uses 1536.
        rates: transposed-convolution strides; their product is the number of
            output samples per input frame (960 = 48 kHz / 50 Hz).
        d_out: output channels (1, mono).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        channels: int = 1536,
        rates: List[int] = [8, 5, 4, 3, 2],
        d_out: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.upsample_factor = int(math.prod(rates))
        layers: List[nn.Module] = [
            _wn_conv1d(input_dim, channels, kernel_size=7, padding=3)
        ]
        output_dim = channels
        for i, stride in enumerate(rates):
            input_dim_i = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim_i, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            _wn_conv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, T_frames) -> (B, 1, T_frames * upsample_factor).

        Same calling convention as the official TorchScript decoder, so the
        two are interchangeable wherever one is used.
        """
        return self.model(x)

    def generate(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        """(B, T_frames, D) features -> (B, T_wav) waveform."""
        return self.forward(ssl_feat.transpose(1, 2)).squeeze(1)

    def remove_weight_norm(self) -> None:
        """Fold weight norm into plain weights for inference."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass  # already folded

    def _ordered_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Parameters in forward-execution order, (weight, bias) per conv."""
        ordered = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                ordered.append((f"{name}.weight", module.weight))
                ordered.append((f"{name}.bias", module.bias))
            elif isinstance(module, Snake1d):
                ordered.append((f"{name}.alpha", module.alpha))
        return ordered

    @staticmethod
    def _official_weights(scripted: torch.jit.ScriptModule) -> List[torch.Tensor]:
        """Weights of a frozen decoder graph, in execution order.

        Freezing folds weight norm and turns every parameter into a graph
        constant, and the constants are not stored in any useful order. The
        graph nodes are, so each convolution and Snake is read off the op
        that consumes it: conv weight and bias from the convolution call,
        alpha from the multiply whose other operand is the activation input
        (the second multiply of a Snake takes 1/(alpha + 1e-9), a derived
        constant that is not a parameter and is skipped).
        """

        def const(value):
            node = value.node()
            return node.t("value") if node.kind() == "prim::Constant" else None

        weights = []
        for node in scripted.graph.nodes():
            kind = node.kind()
            if kind in ("aten::conv1d", "aten::conv_transpose1d", "aten::_convolution"):
                weights.append(const(node.inputsAt(1)))
                weights.append(const(node.inputsAt(2)))
            elif kind == "aten::mul":
                inputs = list(node.inputs())
                consts = [const(v) for v in inputs]
                others = [v for v, c in zip(inputs, consts) if c is None]
                if len(others) != 1 or others[0].node().kind() != "aten::reshape":
                    continue
                weights.append(next(c for c in consts if c is not None))
        return weights

    @torch.no_grad()
    def load_official_torchscript(self, path: str, verify: bool = True) -> None:
        """Load the published ``decoder_{cpu,cuda}.pt`` into this module.

        The release is a *frozen* TorchScript graph with no state_dict; see
        ``_official_weights`` for how its parameters are recovered. Weight
        norm is removed here first, since the graph holds folded weights. By
        default the result is verified end to end by running both decoders
        on the same input.
        """
        scripted = torch.jit.load(path, map_location="cpu")
        weights = self._official_weights(scripted)
        self.remove_weight_norm()
        params = self._ordered_parameters()
        if len(weights) != len(params):
            raise RuntimeError(
                f"{path} holds {len(weights)} weight tensors but this vocoder "
                f"has {len(params)} parameters; the architectures differ "
                f"(check channels/rates in vocoder_conf)."
            )
        for (name, param), value in zip(params, weights):
            if tuple(param.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"shape mismatch at {name}: module {tuple(param.shape)} vs "
                    f"official {tuple(value.shape)}"
                )
            param.copy_(value.to(param.dtype))
        if verify:
            torch.manual_seed(0)
            x = torch.randn(1, self.input_dim, 25)
            was_training = self.training
            self.eval()
            device = next(self.parameters()).device
            ours = self.forward(x.to(device)).float().cpu()
            theirs = scripted(x).float().cpu()
            self.train(was_training)
            err = (ours - theirs).abs().max().item()
            if err > 1e-4:
                raise RuntimeError(
                    f"loaded weights but outputs differ from {path} "
                    f"(max abs error {err:.2e}); parameter order mismatch"
                )


class HiFiGANVocoder(nn.Module):
    """HiFi-GAN generator (ESPnet's own) driven by SSL features.

    The alternative vocoder of the recipe: ESPnet's ``HiFiGANGenerator`` with
    the same 8-5-4-3-2 upsampling geometry as the DAC decoder, so it also
    turns one 50 Hz frame into 960 samples at 48 kHz, but with HiFi-GAN v1
    residual blocks and LeakyReLU instead of Snake. About 14M parameters at
    512 channels. Same calling convention as ``DACVocoder``.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        channels: int = 512,
        kernel_size: int = 7,
        upsample_scales: List[int] = [8, 5, 4, 3, 2],
        upsample_kernel_sizes: List[int] = [16, 10, 8, 6, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Optional[Dict] = None,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        from espnet2.gan_tts.hifigan import HiFiGANGenerator

        self.input_dim = input_dim
        self.upsample_factor = int(math.prod(upsample_scales))
        self.generator = HiFiGANGenerator(
            in_channels=input_dim,
            out_channels=1,
            channels=channels,
            kernel_size=kernel_size,
            upsample_scales=upsample_scales,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params
            or {"negative_slope": 0.1},
            use_weight_norm=use_weight_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, T_frames) -> (B, 1, T_frames * upsample_factor)."""
        return self.generator(x)

    def generate(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        return self.forward(ssl_feat.transpose(1, 2)).squeeze(1)

    def remove_weight_norm(self) -> None:
        self.generator.remove_weight_norm()


# Both train adversarially in RestorationVocoderTask and share the inference path.
VOCODERS = {"dac": DACVocoder, "hifigan": HiFiGANVocoder}


def build_vocoder(
    vocoder_type: str, input_dim: int, conf: Optional[Dict] = None
) -> nn.Module:
    """Instantiate the vocoder named by ``--vocoder_type`` with ``--vocoder_conf``."""
    if vocoder_type not in VOCODERS:
        raise ValueError(
            f"vocoder_type must be one of {sorted(VOCODERS)}, got {vocoder_type!r}"
        )
    return VOCODERS[vocoder_type](input_dim=input_dim, **dict(conf or {}))
