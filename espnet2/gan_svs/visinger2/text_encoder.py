import torch
import math
from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from typing import Optional


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        vocabs: int,
        out_channels: int = 192,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 6,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 3,
        positional_encoding_layer_type: str = "rel_pos",
        self_attention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        normalize_before: bool = True,
        use_macaron_style: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 7,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.emb_phone = torch.nn.Embedding(vocabs, 256)
        torch.nn.init.normal_(self.emb_phone.weight, 0.0, 256**-0.5)

        # self.emb_pitch = torch.nn.Embedding(73, 128)
        self.emb_pitch = torch.nn.Embedding(128, 128)
        torch.nn.init.normal_(self.emb_pitch.weight, 0.0, 128**-0.5)

        # self.emb_slur = torch.nn.Embedding(len(ttsing_slur_set), 64)
        # torch.nn.init.normal_(self.emb_slur.weight, 0.0, 64**-0.5)

        # self.emb_dur = torch.nn.Linear(1, 64)
        self.emb_dur = torch.nn.Linear(1, 128)

        self.pre_net = torch.nn.Linear(512, attention_dim)
        self.pre_dur_net = torch.nn.Linear(512, attention_dim)

        self.encoder = Encoder(
            idim=-1,
            input_layer=None,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            normalize_before=normalize_before,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=positional_encoding_layer_type,
            selfattention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_conformer_conv,
            cnn_module_kernel=conformer_kernel_size,
        )
        self.proj = torch.nn.Conv1d(attention_dim, out_channels, 1)
        self.proj_pitch = torch.nn.Conv1d(128, out_channels, 1)

    def forward(
        self, phone, phone_lengths, pitchid, dur, slur: Optional[torch.Tensor] = None
    ):

        phone_end = self.emb_phone(phone) * math.sqrt(256)
        pitch_end = self.emb_pitch(pitchid) * math.sqrt(128)
        if slur:
            slur_end = self.emb_slur(slur) * math.sqrt(64)
        dur = dur.float()
        dur_end = self.emb_dur(dur.unsqueeze(-1))
        if slur:
            x = torch.cat([phone_end, pitch_end, slur_end, dur_end], dim=-1)
        else:
            x = torch.cat([phone_end, pitch_end, dur_end], dim=-1)

        dur_input = self.pre_dur_net(x)
        dur_input = torch.transpose(dur_input, 1, -1)

        x = self.pre_net(x)
        # x = torch.transpose(x, 1, -1)  # [b, h, t]

        x_mask = (
            make_non_pad_mask(phone_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )

        x, _ = self.encoder(x, x_mask)
        x = x.transpose(1, 2)
        x = self.proj(x) * x_mask

        pitch_info = self.proj_pitch(pitch_end.transpose(1, 2))

        return x, x_mask, dur_input, pitch_info
