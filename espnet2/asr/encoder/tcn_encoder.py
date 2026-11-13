"""Temporal convolutional network (TCN) encoder with a switchable look-ahead.

The network is the Conv-TasNet TCN (Luo and Mesgarani, 2019): a stack of
blocks, each a 1x1 convolution, a dilated depthwise convolution and a 1x1
convolution, with a residual connection, at dilations 1, 2, ..., 2**(X - 1)
repeated R times. The frame rate is kept.

Each depthwise convolution has ``kernel_size`` taps up to the current frame
(the current frame and kernel_size - 1 past taps) and up to ``future_taps``
taps after it. All taps are trained weights of one kernel; the look-ahead only
decides how many of the future taps each layer uses. With look-ahead 0 no
future tap is used and every output frame depends on the current and past input
frames only (online). With look-ahead ``max_look_ahead`` every layer uses all
its future taps (offline). This is the dual-mode convolution of Yu et al.,
"Dual-mode ASR: Unify and Improve Streaming ASR with Full-context Modeling",
ICLR 2021.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_look_ahead_encoder import AbsLookAheadEncoder
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm, choose_norm

# Normalizations over the whole sequence (only for configurations of the
# previous encoder with causal: false; see TCNEncoder).
_TIME_POOLING_NORMS = ("gLN", "GN")


def _norm(norm_type: str, channels: int) -> torch.nn.Module:
    # cLN and BN (with its running statistics at inference) are per frame, so
    # neither reads other frames. gLN and GN, as in the previous encoder,
    # normalize over all frames.
    if norm_type == "cLN":
        return ChannelwiseLayerNorm(channels)
    if norm_type == "BN":
        return torch.nn.BatchNorm1d(channels)
    if norm_type in _TIME_POOLING_NORMS:
        return choose_norm(norm_type, channels)
    raise ValueError(f"norm_type must be 'cLN', 'BN', 'gLN' or 'GN', got {norm_type!r}")


class _TCNBlock(torch.nn.Module):
    """1x1 conv, PReLU, norm, dilated depthwise conv, PReLU, norm, 1x1 conv."""

    def __init__(
        self,
        bottleneck_size: int,
        hidden_size: int,
        kernel_size: int,
        future_taps: int,
        dilation: int,
        norm_type: str,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.max_future_taps = future_taps
        self.future_taps = 0
        self.conv_in = torch.nn.Conv1d(bottleneck_size, hidden_size, 1, bias=False)
        self.act_in = torch.nn.PReLU()
        self.norm_in = _norm(norm_type, hidden_size)
        # Tap k of the kernel reads frame t + (k - kernel_size + 1) * dilation.
        self.depthwise = torch.nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size + future_taps,
            groups=hidden_size,
            bias=False,
        )
        self.act_out = torch.nn.PReLU()
        self.norm_out = _norm(norm_type, hidden_size)
        self.conv_out = torch.nn.Conv1d(hidden_size, bottleneck_size, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the residual for x (N, B, T); mask (N, 1, T) is 1 on valid frames."""
        y = self.norm_in(self.act_in(self.conv_in(x)))  # (N, H, T)
        # Zero the padded frames, so that a padded utterance sees zeros after
        # its end exactly as it does alone.
        y = y * mask
        d = self.dilation
        # (H, 1, kernel_size + future_taps): the taps this look-ahead uses
        weight = self.depthwise.weight[..., : self.kernel_size + self.future_taps]
        # (N, H, (kernel_size - 1) d + T + future_taps d)
        y = F.pad(y, ((self.kernel_size - 1) * d, self.future_taps * d))
        y = F.conv1d(y, weight, dilation=d, groups=y.size(1))  # (N, H, T)
        y = self.norm_out(self.act_out(y))
        return self.conv_out(y)  # (N, B, T)

    def forward_streaming(
        self, x: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the online (no future tap) residual of the new frames x (N, B, k).

        cache (N, H, (kernel_size - 1) * dilation) holds the depthwise
        convolution's inputs of the frames before x (zeros before the first
        frame, as the left padding of forward()). Returns the residual and the
        cache for the next call.
        """
        y = self.norm_in(self.act_in(self.conv_in(x)))  # (N, H, k)
        y = torch.cat([cache, y], dim=2)  # (N, H, (kernel_size - 1) d + k)
        weight = self.depthwise.weight[..., : self.kernel_size]  # causal taps
        out = F.conv1d(y, weight, dilation=self.dilation, groups=y.size(1))  # (N, H, k)
        cache = y[..., y.size(2) - cache.size(2) :]  # (N, H, (kernel_size - 1) d)
        return self.conv_out(self.norm_out(self.act_out(out))), cache  # (N, B, k)


class TCNEncoder(AbsLookAheadEncoder):
    """Conv-TasNet TCN encoder whose look-ahead is chosen at run time.

    Args:
        input_size: Input feature dimension.
        output_size: Output dimension.
        bottleneck_size: Channels between the blocks (B).
        hidden_size: Channels inside each block (H).
        kernel_size: Depthwise taps up to the current frame (P): the current
            frame and P - 1 past taps.
        num_blocks: Blocks per repeat, at dilations 1, 2, ..., 2**(X - 1) (X).
        num_repeats: Number of repeats (R).
        future_taps: Depthwise taps after the current frame that a layer can
            use. 0 makes a causal encoder. Keyword-only, like look_ahead.
        look_ahead: Look-ahead in frames to start with; set_look_ahead()
            changes it.
        norm_type: "cLN" (layer norm over channels, per frame) or "BN". "gLN"
            and "GN" (over all frames) only with causal=False, for
            configurations of the previous encoder: every output then reads
            the whole sequence, so the encoder has no bounded look-ahead (the
            look-ahead cannot be changed, receptive_field and
            forward_streaming raise) and a padded utterance in a batch is
            normalized with the padding, as in the previous encoder.
        pre_output_nonlinear: Nonlinearity before the output projection:
            "linear", "prelu", "relu", "tanh" or "sigmoid".
        causal: The configuration of the previous TCNEncoder (a wrapper around
            espnet2.enh.layers.tcn.TemporalConvNet), where kernel_size counted
            all the taps. None: not used. True: future_taps 0 (the same
            network). False: the centred kernel of kernel_size taps (odd), i.e.
            (kernel_size + 1) // 2 taps up to the current frame and
            (kernel_size - 1) // 2 after it, at the full look-ahead. Give it
            without future_taps and look_ahead. Checkpoints of the previous
            encoder load: their state-dict keys are renamed.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 128,
        bottleneck_size: int = 64,
        hidden_size: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 5,
        num_repeats: int = 2,
        norm_type: str = "cLN",
        causal: Optional[bool] = None,
        pre_output_nonlinear: str = "prelu",
        *,
        future_taps: int = 0,
        look_ahead: int = 0,
    ):
        super().__init__()
        if causal is not None:
            if future_taps != 0 or look_ahead != 0:
                raise ValueError(
                    "give causal (the previous TCNEncoder's option) or "
                    "future_taps/look_ahead, not both"
                )
            if not causal:
                if kernel_size % 2 == 0:
                    raise ValueError(
                        f"causal=False needs an odd kernel_size, got {kernel_size}"
                    )
                future_taps = (kernel_size - 1) // 2
                kernel_size = (kernel_size + 1) // 2
        self.pools_over_time = norm_type in _TIME_POOLING_NORMS
        if self.pools_over_time and causal is not False:
            raise ValueError(
                f"norm_type {norm_type!r} normalizes over all frames, so every "
                "output reads the whole sequence: it is only accepted with "
                "causal=False (configurations of the previous encoder); use "
                "'cLN' or 'BN' for an encoder with a bounded look-ahead"
            )
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be at least 1, got {kernel_size}")
        if future_taps < 0:
            raise ValueError(f"future_taps must be at least 0, got {future_taps}")
        self._output_size = output_size
        self.input_norm = ChannelwiseLayerNorm(input_size)
        self.bottleneck = torch.nn.Conv1d(input_size, bottleneck_size, 1, bias=False)
        self.blocks = torch.nn.ModuleList(
            _TCNBlock(
                bottleneck_size,
                hidden_size,
                kernel_size,
                future_taps,
                2**x,
                norm_type,
            )
            for _ in range(num_repeats)
            for x in range(num_blocks)
        )
        self.output_act = {
            "linear": torch.nn.Identity(),
            "prelu": torch.nn.PReLU(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
        }[pre_output_nonlinear]
        self.output = torch.nn.Conv1d(bottleneck_size, output_size, 1, bias=False)
        self._look_ahead = 0
        self.set_look_ahead(self.max_look_ahead if causal is False else look_ahead)
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args: self.upgrade_state_dict(
                state_dict, prefix
            )
        )

    def upgrade_state_dict(self, state_dict: dict, prefix: str = ""):
        """Convert a state dict of the previous TCNEncoder in place.

        That encoder held espnet2.enh.layers.tcn.TemporalConvNet as ``tcn``:
        network.0 input norm, network.1 bottleneck, network.2.<r>.<x> blocks
        (net.0 1x1 conv, net.1 PReLU, net.2 norm, net.3.net depthwise conv,
        [Chomp1d when causal,] PReLU, norm, 1x1 conv), then the output, after a
        nonlinearity module unless it was linear. Its keys are renamed, and a
        depthwise kernel with fewer taps than this encoder's gets zero future
        taps appended: a causal checkpoint then initializes an encoder with
        future taps and gives the same output at every look-ahead until the
        future taps are trained. load_state_dict() calls it; call it directly
        before code that compares keys or shapes first (e.g.
        espnet2.torch_utils.load_pretrained_model). Other state dicts are left
        as they are.
        """
        old = prefix + "tcn.network."
        keys = [k for k in state_dict if k.startswith(old)]
        if not keys:
            return
        rests = [k[len(old) :] for k in keys]
        has_output_act = any(r.startswith("4.") for r in rests)
        # the causal blocks have a Chomp1d (no parameters) at net.3.net.1
        chomp = any(r.split(".")[5:7] == ["net", "4"] for r in rests)
        dsconv = (
            {"0": "depthwise", "2": "act_out", "3": "norm_out", "4": "conv_out"}
            if chomp
            else {"0": "depthwise", "1": "act_out", "2": "norm_out", "3": "conv_out"}
        )
        block_part = {"0": "conv_in", "1": "act_in", "2": "norm_in"}
        num_blocks = len({b.dilation for b in self.blocks})
        for key, rest in zip(keys, rests):
            parts = rest.split(".")
            if parts[0] == "0":
                new = ["input_norm"] + parts[1:]
            elif parts[0] == "1":
                new = ["bottleneck"] + parts[1:]
            elif parts[0] == "2":
                # 2.<r>.<x>.net.<i>.<param> or 2.<r>.<x>.net.3.net.<j>.<param>
                i = int(parts[1]) * num_blocks + int(parts[2])
                if parts[4] == "3":
                    new = ["blocks", str(i), dsconv[parts[6]]] + parts[7:]
                else:
                    new = ["blocks", str(i), block_part[parts[4]]] + parts[5:]
            elif parts[0] == "3" and has_output_act:
                new = ["output_act"] + parts[1:]
            else:  # the output convolution: network.3, or network.4 after a module
                new = ["output"] + parts[1:]
            value = state_dict.pop(key)
            if new[0] == "blocks" and new[2] == "depthwise":
                taps = self.blocks[int(new[1])].depthwise.weight.size(-1)
                if value.size(-1) < taps:  # append zero future taps
                    value = torch.nn.functional.pad(value, (0, taps - value.size(-1)))
            state_dict[prefix + ".".join(new)] = value

    def output_size(self) -> int:
        return self._output_size

    @property
    def past_context(self) -> int:
        """Number of past input frames each output frame reads."""
        return sum((b.kernel_size - 1) * b.dilation for b in self.blocks)

    @property
    def receptive_field(self) -> int:
        """Number of input frames an output frame reads at look-ahead 0.

        The current frame and past_context frames before it. A streaming caller
        can rerun the encoder on each chunk's new frames together with the
        receptive_field - 1 frames before them and get the whole-recording
        output for the new frames.

        Raises:
            RuntimeError: With gLN/GN, where every output reads every frame.
        """
        if self.pools_over_time:
            raise RuntimeError("with gLN/GN every output reads the whole sequence")
        return self.past_context + 1

    @property
    def max_look_ahead(self) -> int:
        return sum(b.max_future_taps * b.dilation for b in self.blocks)

    @property
    def look_ahead(self) -> int:
        return self._look_ahead

    def future_taps_per_block(self) -> List[int]:
        """Number of future taps each block uses at the current look-ahead."""
        return [b.future_taps for b in self.blocks]

    def set_look_ahead(self, frames: int):
        """Make each output frame t read input frames up to t + frames.

        The look-ahead goes to the blocks with the largest dilation first: a
        block takes as many future taps as fit, up to future_taps, and the rest
        goes to the next one.

        Raises:
            ValueError: frames is negative, larger than max_look_ahead, or
                cannot be made from the blocks' future taps.
        """
        if self.pools_over_time and frames != self.max_look_ahead:
            raise ValueError(
                "with gLN/GN every output reads the whole sequence: the "
                f"look-ahead stays at the full kernel ({self.max_look_ahead})"
            )
        if not 0 <= frames <= self.max_look_ahead:
            raise ValueError(
                f"look_ahead must be in [0, {self.max_look_ahead}], got {frames}"
            )
        taps = {}
        left = frames
        order = sorted(range(len(self.blocks)), key=lambda i: -self.blocks[i].dilation)
        for i in order:
            block = self.blocks[i]
            taps[i] = min(block.max_future_taps, left // block.dilation)
            left -= taps[i] * block.dilation
        if left != 0:
            raise ValueError(
                f"look_ahead {frames} cannot be made from the future taps of the "
                f"blocks (dilations {[b.dilation for b in self.blocks]})"
            )
        for i, block in enumerate(self.blocks):
            block.future_taps = taps[i]
        self._look_ahead = frames

    def forward_streaming(
        self, xs: torch.Tensor, state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode the next frames of a stream online (look-ahead 0).

        Feeding a sequence in chunks of any sizes gives forward()'s output on
        the whole sequence at look-ahead 0, whatever look-ahead is set. With
        norm_type "BN" the encoder must be in eval mode. Every
        layer but the depthwise convolutions works per frame, so the state is
        the last (kernel_size - 1) * dilation depthwise inputs of each block:
        receptive_field - 1 frames in all, whatever the stream length.

        Args:
            xs: New input frames (B, k, input_size), no padding.
            state: The state returned by the previous call; None at the start
                of a stream.

        Returns:
            Encoded new frames (B, k, output_size) and the state for the next
            call.
        """
        if self.pools_over_time:
            raise RuntimeError(
                "forward_streaming needs per-frame normalization (cLN or BN): "
                "gLN/GN normalize over the whole sequence"
            )
        if self.training and any(
            isinstance(m, torch.nn.BatchNorm1d) for m in self.modules()
        ):
            raise RuntimeError(
                "forward_streaming with BatchNorm needs eval mode: in training "
                "mode BatchNorm normalizes each chunk with its own statistics"
            )
        x = self.bottleneck(self.input_norm(xs.transpose(1, 2)))
        if state is None:
            state = [
                x.new_zeros(
                    x.size(0),
                    b.depthwise.in_channels,
                    (b.kernel_size - 1) * b.dilation,
                )
                for b in self.blocks
            ]
        new_state = []
        for block, cache in zip(self.blocks, state):
            residual, cache = block.forward_streaming(x, cache)
            x = x + residual
            new_state.append(cache)
        out = self.output(self.output_act(x)).transpose(1, 2)
        return out, new_state

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode a padded batch.

        Args:
            xs_pad: Input features (B, T, input_size).
            ilens: Input lengths (B,).
            prev_states: Not used.

        Returns:
            Encoded features (B, T, output_size), output lengths (B,), None.
        """
        x = xs_pad.transpose(1, 2)  # (B, input_size, T)
        # 1 on the frames before each length; built on the device (no host sync)
        frames = torch.arange(x.size(2), device=x.device)  # (T,)
        mask = (frames < ilens.to(x.device)[:, None]).unsqueeze(1)  # (B, 1, T)
        mask = mask.to(x.dtype)
        x = self.bottleneck(self.input_norm(x)) * mask  # (B, bottleneck_size, T)
        for block in self.blocks:
            x = (x + block(x, mask)) * mask  # (B, bottleneck_size, T)
        out = self.output(self.output_act(x))  # (B, output_size, T)
        out = out.transpose(1, 2) * mask.transpose(1, 2)  # (B, T, output_size)
        return out, ilens, None
