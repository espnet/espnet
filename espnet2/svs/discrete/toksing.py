# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Renmin University of China (Shuai Guo)
# Copyright 2024 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""XiaoiceSing related modules."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.discrete.loss import DiscreteLoss
from espnet2.svs.xiaoice.loss import XiaoiceSing2Loss
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet.nets.pytorch_backend.conformer.encoder import (  # noqa: H301
    Encoder as ConformerEncoder,
)
from espnet.nets.pytorch_backend.e2e_tts_fastspeech import (
    FeedForwardTransformerLoss as XiaoiceSingLoss,  # NOQA
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import (  # noqa: H301
    Encoder as TransformerEncoder,
)

# from espnet2.gan_svs.vits.pitch_predictor import Decoder

# class Discrete_Postnet(torch.nn.Module):
#     def __init__(
#         self,
#     )


class Decoder(torch.nn.Module):
    """Pitch or Mel decoder module in VISinger 2."""

    def __init__(
        self,
        out_channels: int = 192,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 6,
        pw_layer_type: str = "conv1d",
        pw_conv_kernel_size: int = 3,
        pos_enc_layer_type: str = "rel_pos",
        self_attention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        normalize_before: bool = True,
        use_macaron_style: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 7,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        global_channels: int = -1,
    ):
        """
        Args:
            out_channels (int): The output dimension of the module.
            attention_dim (int): The dimension of the attention mechanism.
            attention_heads (int): The number of attention heads.
            linear_units (int): The number of units in the linear layer.
            blocks (int): The number of encoder blocks.
            pw_layer_type (str): The type of position-wise layer to use.
            pw_conv_kernel_size (int): The kernel size of the position-wise
                                       convolutional layer.
            pos_enc_layer_type (str): The type of positional encoding layer to use.
            self_attention_layer_type (str): The type of self-attention layer to use.
            activation_type (str): The type of activation function to use.
            normalize_before (bool): Whether to normalize the data before the
                                     position-wise layer or after.
            use_macaron_style (bool): Whether to use the macaron style or not.
            use_conformer_conv (bool): Whether to use Conformer style conv or not.
            conformer_kernel_size (int): The kernel size of the conformer
                                         convolutional layer.
            dropout_rate (float): The dropout rate to use.
            positional_dropout_rate (float): The positional dropout rate to use.
            attention_dropout_rate (float): The attention dropout rate to use.
            global_channels (int): The number of channels to use for global
                                   conditioning.
        """
        super().__init__()

        self.prenet = torch.nn.Conv1d(attention_dim, attention_dim, 3, padding=1)
        self.decoder = ConformerEncoder(
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
            positionwise_layer_type=pw_layer_type,
            positionwise_conv_kernel_size=pw_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=pos_enc_layer_type,
            selfattention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_conformer_conv,
            cnn_module_kernel=conformer_kernel_size,
        )
        self.proj = torch.nn.Conv1d(attention_dim, out_channels, 1)

        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, attention_dim, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Forward pass of the Decoder.

        Args:
            x (Tensor): Input tensor (B, 2 + attention_dim, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Tensor, optional): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, 1, T).
            Tensor: Output mask (B, 1, T).
        """

        x_mask = (
            make_non_pad_mask(x_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )

        x = self.prenet(x) * x_mask

        if g is not None:
            x = x + self.global_conv(g)

        x = x.transpose(1, 2)
        x, _ = self.decoder(x, x_mask)
        x = x.transpose(1, 2)

        x = self.proj(x) * x_mask

        return x, x_mask


class TokSing(AbsSVS):
    """ TokSing: Singing Voice Synthesis based on Discrete Tokens

        paper link: https://arxiv.org/abs/2406.08416
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        midi_dim: int = 129,
        duration_dim: int = 500,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # lf0 prediction related
        global_channels: int = -1,
        text_encoder_attention_heads: int = 2,
        text_encoder_ffn_expand: int = 4,
        text_encoder_blocks: int = 6,
        text_encoder_positionwise_layer_type: str = "conv1d",
        text_encoder_positionwise_conv_kernel_size: int = 1,
        text_encoder_positional_encoding_layer_type: str = "rel_pos",
        text_encoder_self_attention_layer_type: str = "rel_selfattn",
        text_encoder_activation_type: str = "swish",
        text_encoder_normalize_before: bool = True,
        text_encoder_dropout_rate: float = 0.1,
        text_encoder_positional_dropout_rate: float = 0.0,
        text_encoder_attention_dropout_rate: float = 0.0,
        text_encoder_conformer_kernel_size: int = 7,
        use_macaron_style_in_text_encoder: bool = True,
        use_conformer_conv_in_text_encoder: bool = True,
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_function: str = "XiaoiceSing2",  # FastSpeech1, XiaoiceSing2
        loss_type: str = "L1",
        lambda_out: float = 1,
        lambda_dur: float = 0.1,
        lambda_pitch: float = 0.01,
        lambda_vuv: float = 0.01,
        use_discrete_token: bool = False,
        discrete_token_layers: int = 1,
        predict_pitch: bool = False,
        codec_codebook: int = 0,
    ):
        """Initialize XiaoiceSing module.

        Args:
            idim (int): Dimension of the label inputs.
            odim (int): Dimension of the outputs.
            midi_dim (int): Dimension of the midi inputs.
            duration_dim (int): Dimension of the duration inputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Kernel size of postnet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            decoder_type (str): Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.
            loss_function (str): Loss functions ("FastSpeech1" or "XiaoiceSing2")
            loss_type (str): Mel loss type ("L1" (MAE), "L2" (MSE) or "L1+L2")
            lambda_out (float): Loss scaling coefficient for Mel or discrete token loss.
            lambda_dur (float): Loss scaling coefficient for duration loss.
            lambda_pitch (float): Loss scaling coefficient for pitch loss.
            lambda_vuv (float): Loss scaling coefficient for VUV loss.
            use_discrete_token (bool): Whether to use discrete tokens as targets.
            predict_pitch (bool): Whether to predict pitch when use_discrete_token.

        """
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.duration_dim = duration_dim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_function = loss_function
        self.loss_type = loss_type
        self.lambda_out = lambda_out
        self.lambda_dur = lambda_dur
        self.lambda_pitch = lambda_pitch
        self.lambda_vuv = lambda_vuv
        self.use_discrete_token = use_discrete_token
        self.discrete_token_layers = discrete_token_layers
        self.proj_out_dim = self.odim * self.discrete_token_layers
        self.predict_pitch = predict_pitch
        self.codec_codebook = codec_codebook

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # check relative positional encoding compatibility
        if "conformer" in [encoder_type, decoder_type]:
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        self.phone_encode_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        self.midi_encode_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=adim,
            padding_idx=self.padding_idx,
        )
        self.duration_encode_layer = torch.nn.Embedding(
            num_embeddings=duration_dim,
            embedding_dim=adim,
            padding_idx=self.padding_idx,
        )

        if self.use_discrete_token and self.predict_pitch:
            self.proj_pitch = torch.nn.Linear(adim, adim)
            self.f0_predictor = Decoder(
                1,
                attention_dim=adim,
                attention_heads=text_encoder_attention_heads,
                linear_units=adim * text_encoder_ffn_expand,
                blocks=text_encoder_blocks,
                pw_layer_type=text_encoder_positionwise_layer_type,
                pw_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
                pos_enc_layer_type=text_encoder_positional_encoding_layer_type,
                self_attention_layer_type=text_encoder_self_attention_layer_type,
                activation_type=text_encoder_activation_type,
                normalize_before=text_encoder_normalize_before,
                dropout_rate=text_encoder_dropout_rate,
                positional_dropout_rate=text_encoder_positional_dropout_rate,
                attention_dropout_rate=text_encoder_attention_dropout_rate,
                conformer_kernel_size=text_encoder_conformer_kernel_size,
                use_macaron_style=use_macaron_style_in_text_encoder,
                use_conformer_conv=use_conformer_conv_in_text_encoder,
                global_channels=global_channels,
            )

        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu,
            )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        sep = False
        self.sep = sep
        if decoder_type == "transformer":
            if self.discrete_token_layers > 1 and self.sep:
                self.decoder = torch.nn.ModuleList(
                    [
                        TransformerEncoder(
                            idim=0,
                            attention_dim=adim,
                            attention_heads=aheads,
                            linear_units=dunits,
                            num_blocks=dlayers,
                            input_layer=None,
                            dropout_rate=transformer_dec_dropout_rate,
                            positional_dropout_rate=transformer_dec_positional_dropout_rate,
                            attention_dropout_rate=transformer_dec_attn_dropout_rate,
                            pos_enc_class=pos_enc_class,
                            normalize_before=decoder_normalize_before,
                            concat_after=decoder_concat_after,
                            positionwise_layer_type=positionwise_layer_type,
                            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                        )
                        for i in range(self.discrete_token_layers)
                    ]
                )
            else:
                self.decoder = TransformerEncoder(
                    idim=0,
                    attention_dim=adim,
                    attention_heads=aheads,
                    linear_units=dunits,
                    num_blocks=dlayers,
                    input_layer=None,
                    dropout_rate=transformer_dec_dropout_rate,
                    positional_dropout_rate=transformer_dec_positional_dropout_rate,
                    attention_dropout_rate=transformer_dec_attn_dropout_rate,
                    pos_enc_class=pos_enc_class,
                    normalize_before=decoder_normalize_before,
                    concat_after=decoder_concat_after,
                    positionwise_layer_type=positionwise_layer_type,
                    positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                )
        elif decoder_type == "conformer":
            self.decoder = ConformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_dec_kernel_size,
            )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        if self.discrete_token_layers > 1 and self.sep:
            self.linear_projection = torch.nn.ModuleList(
                [
                    torch.nn.Linear(adim, self.odim * reduction_factor)
                    for i in range(self.discrete_token_layers)
                ]
            )
        else:
            self.linear_projection = torch.nn.Linear(
                adim, self.proj_out_dim * reduction_factor
            )
        if self.loss_function == "XiaoiceSing2" or self.predict_pitch:
            # self.pitch_predictor = torch.nn.Linear(adim, 1 * reduction_factor)
            self.lf0_mapping = torch.nn.Linear(
                1 * reduction_factor, adim * reduction_factor
            )
        if self.loss_function == "XiaoiceSing2":
            self.vuv_predictor = torch.nn.Linear(adim, 1 * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        discrete_postnet_layers = 0
        self.discrete_postnet_layers = discrete_postnet_layers
        if discrete_postnet_layers != 0:
            dplayers = discrete_postnet_layers
            dpunits = 1536
            transformer_dpos_dropout_rate = 0.1
            transformer_dpos_positional_dropout_rate = 0.1
            transformer_dpos_attn_dropout_rate = 0.1
            pos_dpos_class = (
                ScaledPositionalEncoding
                if self.use_scaled_pos_enc
                else PositionalEncoding
            )
            dpostnet_normalize_before = True
            dpostnet_concat_after = False
            token_dim = self.odim

            if self.discrete_token_layers == 1:
                self.token_emb = torch.nn.Embedding(
                    num_embeddings=token_dim,
                    embedding_dim=adim,
                    padding_idx=self.padding_idx,
                )
            else:
                self.token_emb = [
                    torch.nn.Embedding(
                        num_embeddings=token_dim,
                        embedding_dim=adim,
                        padding_idx=self.padding_idx,
                    )
                    for i in range(self.discrete_token_layers)
                ]

            self.discrete_postnet = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dpunits,
                num_blocks=dplayers,
                input_layer=None,
                dropout_rate=transformer_dpos_dropout_rate,
                positional_dropout_rate=transformer_dpos_positional_dropout_rate,
                attention_dropout_rate=transformer_dpos_attn_dropout_rate,
                pos_enc_class=pos_dpos_class,
                normalize_before=dpostnet_normalize_before,
                concat_after=dpostnet_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
            self.dpos_linear_projection = torch.nn.Linear(
                adim, self.odim * reduction_factor
            )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterions
        if self.use_discrete_token:
            self.criterion = DiscreteLoss(
                use_masking=use_masking,
                use_weighted_masking=use_weighted_masking,
                predict_pitch=self.predict_pitch,
            )
        else:
            if self.loss_function == "FastSpeech1":
                self.criterion = XiaoiceSingLoss(
                    use_masking=use_masking, use_weighted_masking=use_weighted_masking
                )
            elif self.loss_function == "XiaoiceSing2":
                self.criterion = XiaoiceSing2Loss(
                    use_masking=use_masking, use_weighted_masking=use_weighted_masking
                )
            else:
                raise ValueError(f"{self.loss_function} is not supported.")

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        duration_lengths: Optional[Dict[str, torch.Tensor]] = None,
        slur: torch.LongTensor = None,
        slur_lengths: torch.Tensor = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
        discrete_token: torch.Tensor = None,
        discrete_token_lengths: torch.Tensor = None,
        discrete_token_lengths_frame: torch.Tensor = None,
        flag_IsValid: bool = False,
        flag_RL: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_frame, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            melody_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded melody (B, ).
            pitch (FloatTensor): Batch of padded f0 (B, T_frame).
            pitch_lengths (LongTensor): Batch of the lengths of padded f0 (B, ).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            duration_length (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of the lengths of padded duration (B, ).
            slur (LongTensor): Batch of padded slur (B, T_frame).
            slur_lengths (LongTensor): Batch of the lengths of padded slur (B, ).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            discrete_token (LongTensor): Batch of padded discrete tokens (B, T_frame).
            discrete_token_lengths (LongTensor): Batch of the lengths of padded slur (B, ).
            joint_training (bool): Whether to perform joint training with vocoder.
            flag_IsValid (bool): Whether it is valid set.
            falg_RL (bool): Whether to perform reinforcement learning. (RL will use model in infer mode.)

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """

        if joint_training:
            label = label
            midi = melody
            label_lengths = label_lengths
            midi_lengths = melody_lengths
            duration_lengths = duration_lengths
            duration_ = duration
            ds = duration
        else:
            label = label["score"]
            midi = melody["score"]
            duration_ = duration["score_syb"]
            label_lengths = label_lengths["score"]
            midi_lengths = melody_lengths["score"]
            duration_lengths = duration_lengths["lab"]
            ds = duration["lab"]

        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        duration_ = duration_[:, : duration_lengths.max()]  # for data-parallel
        if self.loss_function == "XiaoiceSing2" or self.use_discrete_token:
            log_f0 = pitch[:, : pitch_lengths.max()]
        if self.loss_function == "XiaoiceSing2":
            vuv = log_f0 != 0
        batch_size = text.size(0)

        label_emb = self.phone_encode_layer(label)
        midi_emb = self.midi_encode_layer(midi)
        duration_emb = self.duration_encode_layer(duration_)
        input_emb = label_emb + midi_emb + duration_emb

        x_masks = self._source_mask(label_lengths)
        hs, _ = self.encoder(input_emb, x_masks)  # (B, T_text, adim)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(label_lengths).to(input_emb.device)
        hs = hs.masked_fill(d_masks.unsqueeze(-1), 0.0)
        logging.info(f'd_masks({d_masks.shape}): {d_masks}')
        logging.info(f'ds({ds.shape}): {ds}')

        # NOTE(Yuxun): if using RL, infer mode will be used.
        if flag_RL:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
            d_outs_int = torch.floor(d_outs + 0.5).to(dtype=torch.long)  # (B, T_text)
            logging.info(f'd_outs_int: {d_outs_int} {torch.sum(d_outs_int, dim=1)}')
            hs = self.length_regulator(hs, d_outs_int)
        else:
            d_outs = self.duration_predictor(hs, d_masks)  # (B, T_text)
            hs = self.length_regulator(hs, ds)  # (B, T_feats, adim)

        if self.predict_pitch:
            # NOTE(Yuxun): if using RL, infer mode will be used.
            if not flag_RL:
                hs_pitch_in = self.proj_pitch(midi_emb)
                hs_pitch = self.length_regulator(hs_pitch_in, ds)
                log_f0_outs, _ = self.f0_predictor(
                    (hs + hs_pitch).transpose(1, 2), discrete_token_lengths_frame
                )
                # log_f0_outs, _ = self.f0_predictor(hs.transpose(1, 2), discrete_token_lengths_frame)
                log_f0_outs = log_f0_outs.transpose(1, 2)
                log_f0_outs = torch.max(
                    log_f0_outs, torch.zeros_like(log_f0_outs).to(log_f0_outs)
                )
                hs = hs + self.lf0_mapping(log_f0)
            else:
                hs_pitch_in = self.proj_pitch(midi_emb)
                hs_pitch = self.length_regulator(hs_pitch_in, d_outs_int)
                discrete_token_lengths_frame = torch.Tensor([hs.size(1)]).to(hs.device).to(dtype=torch.long)
                log_f0_outs, _ = self.f0_predictor(
                    (hs + hs_pitch).transpose(1, 2),
                    discrete_token_lengths_frame
                )
                log_f0_outs = log_f0_outs.transpose(1, 2)
                log_f0_outs = torch.max(
                    log_f0_outs, torch.zeros_like(log_f0_outs).to(log_f0_outs)
                )
                hs = hs + self.lf0_mapping(log_f0_outs)

        # forward decoder
        if self.use_discrete_token:
            olens = discrete_token_lengths_frame
        else:
            olens = feats_lengths
        if self.reduction_factor > 1:
            olens_in = olens.new(
                [
                    torch.div(olen, self.reduction_factor, rounding_mode="trunc")
                    for olen in olens
                ]
            )
        else:
            olens_in = olens
        logging.info(f'olen: {olens_in}')
        h_masks = self._source_mask(olens_in)

        if self.discrete_token_layers > 1 and self.sep:
            before_outs = []
            for i in range(self.discrete_token_layers):
                # print(hs.device, h_masks.device, self.decoder[i].device, flush=True)
                zs, _ = self.decoder[i](hs, h_masks)
                before_outs_ = self.linear_projection[i](zs).view(
                    zs.size(0), -1, self.odim
                )
                before_outs.append(before_outs_)
            before_outs = (
                torch.cat(before_outs, dim=2)
                .view(zs.size(0), -1, self.odim)
                .to(ds.device)
            )
        else:
            zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
            # (B. T_feats, odim), (B. T_feats, 1), (B. T_feats, 1)
            before_outs = self.linear_projection(zs).view(
                zs.size(0), -1, self.odim
            )  # (B, T_feats * layers, nclusters)
        # if self.loss_function == "XiaoiceSing2" or self.use_discrete_token:
        #    log_f0_outs = self.pitch_predictor(zs).view(
        #        zs.size(0), -1, 1
        #    )  # (B, T_feats, odim)
        if self.loss_function == "XiaoiceSing2":
            vuv_outs = self.vuv_predictor(zs).view(
                zs.size(0), -1, 1
            )  # (B, T_feats, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        if self.discrete_postnet_layers != 0:
            # print(after_outs.shape, flush=True)
            pred_token = torch.argmax(after_outs, dim=2)
            # print(pred_token.shape, flush=True)
            if self.discrete_token_layers == 1:
                token_emb = self.token_emb(pred_token)
            else:
                token_emb = 0
                for i in range(self.discrete_token_layers):
                    token_emb = token_emb + self.token_emb[i](
                        pred_token[:, i :: self.discrete_token_layers, :]
                    )
            # print(token_emb.shape, flush=True)
            dpos_masks = self._source_mask(olens_in)
            after_outs, _ = self.discrete_postnet(token_emb, dpos_masks)
            after_outs = self.dpos_linear_projection(after_outs)
            # print(after_outs.shape, flush=True)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            assert feats_lengths.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = feats_lengths.new(
                [olen - olen % self.reduction_factor for olen in feats_lengths]
            )
            max_olen = max(olens)
            ys = feats[:, :max_olen]
            if self.loss_function == "XiaoiceSing2":
                log_f0 = log_f0[:, :max_olen]
                vuv = vuv[:, :max_olen]
        else:
            if self.use_discrete_token:
                ys = discrete_token
                if self.codec_codebook > 0:
                    shift = (
                        torch.arange(self.codec_codebook).view(1, 1, -1) * self.odim
                    ).to(ds.device)
                    ys = (
                        discrete_token.view(batch_size, -1, self.codec_codebook) - shift
                    )
                    ys = ys.flatten(start_dim=1)
                olens = discrete_token_lengths
            else:
                ys = feats
                olens = feats_lengths

        ilens = label_lengths
        if self.predict_pitch:
            out_loss, duration_loss, pitch_loss = self.criterion(
                after_outs,
                before_outs,
                d_outs,
                ys,
                ds,
                ilens,
                olens,
                log_f0_outs,
                log_f0,
                pitch_lengths,
            )
        else:
            if self.loss_function == "FastSpeech1":
                out_loss, duration_loss = self.criterion(
                    after_outs, before_outs, d_outs, ys, ds, ilens, olens
                )
            elif self.loss_function == "XiaoiceSing2":
                out_loss, duration_loss, pitch_loss, vuv_loss = self.criterion(
                    after_outs=after_outs,
                    before_outs=before_outs,
                    d_outs=d_outs,
                    p_outs=log_f0_outs,
                    v_outs=vuv_outs,
                    ys=ys,
                    ds=ds,
                    ps=log_f0,
                    vs=vuv,
                    ilens=ilens,
                    olens=olens,
                    loss_type=self.loss_type,
                )

        out_loss = out_loss * self.lambda_out
        duration_loss = duration_loss * self.lambda_dur
        loss = out_loss + duration_loss
        stats = dict(out_loss=out_loss.item(), duration_loss=duration_loss.item())
        if self.loss_function == "XiaoiceSing2" or self.predict_pitch:
            pitch_loss = pitch_loss * self.lambda_pitch
            stats["pitch_loss"] = pitch_loss.item()
            loss += pitch_loss
        if self.loss_function == "XiaoiceSing2":
            vuv_loss = vuv_loss * self.lambda_vuv
            stats["vuv_loss"] = vuv_loss.item()
            loss += vuv_loss
        stats["loss"] = loss.item()

        if self.use_discrete_token:
            gen_token = torch.argmax(after_outs, dim=2)
            token_mask = make_non_pad_mask(discrete_token_lengths).to(ds.device)
            acc = (
                (gen_token == discrete_token) * token_mask
            ).sum().item() / discrete_token_lengths.sum().item()
            stats["acc"] = acc

        # report extra information
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
            )
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        if joint_training:
            return loss, stats, after_outs if after_outs is not None else before_outs
        else:
            if flag_IsValid:
                return loss, stats, weight, after_outs[:, : olens.max()], ys, olens
            elif flag_RL:
                return loss, stats, weight, after_outs[:, : olens.max()]
            else:
                return loss, stats, weight

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        discrete_token: Optional[torch.Tensor] = None,
        use_teacher_forcing: torch.Tensor = False,
        joint_training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (T_frame, idim).
            durations (Optional[LongTensor]): Groundtruth of duration (T_text + 1,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (Tmax).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (Tmax).
            pitch (FloatTensor): Batch of padded f0 (T_frame).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (Tmax).
            slur (LongTensor): Batch of padded slur (T_frame).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            discrete_token (Optional[Tensor]): Batch of discrete tokens (T_frame).

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
        """

        label = label["score"]
        midi = melody["score"]
        if joint_training:
            duration_ = duration["lab"]
        else:
            duration_ = duration["score_syb"]
        ds = duration["lab"]

        label_emb = self.phone_encode_layer(label)
        midi_emb = self.midi_encode_layer(midi)
        duration_emb = self.duration_encode_layer(duration_)
        input_emb = label_emb + midi_emb + duration_emb
        logging.info(f'label({label.shape}): {label}')
        logging.info(f'midi({midi.shape}): {midi}')
        logging.info(f'duration_({duration_.shape}): {duration_}')
        logging.info(f'label_emb({label_emb.shape}): {label_emb}')
        logging.info(f'midi_emb({midi_emb.shape}): {midi_emb}')
        logging.info(f'duration_emb({duration_emb.shape}): {duration_emb}')
        logging.info(f'input_emb({input_emb.shape}): {input_emb}')

        x_masks = None  # self._source_mask(label_lengths)
        hs, _ = self.encoder(input_emb, x_masks)  # (B, T_text, adim)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)
        if spembs is not None:
            spembs = spembs.unsqueeze(0)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        logging.info(f'hs({hs.shape}): {hs}')

        # forward duration predictor and length regulator
        d_masks = None  # make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
        d_outs_int = torch.floor(d_outs + 0.5).to(dtype=torch.long)  # (B, T_text)
        logging.info(f'd_outs_int ({torch.sum(d_outs_int, axis=1)}): {d_outs_int}')

        # use duration model output
        hs = self.length_regulator(hs, d_outs_int)  # (B, T_feats, adim)
        if self.predict_pitch:
            hs_pitch_in = self.proj_pitch(midi_emb)
            hs_pitch = self.length_regulator(hs_pitch_in, d_outs_int)
            # print(hs.shape, hs_pitch.shape)
            # print(torch.Tensor([hs.size(1)]).to(hs.device).to(dtype=torch.long))
            log_f0_outs, _ = self.f0_predictor(
                (hs + hs_pitch).transpose(1, 2),
                torch.Tensor([hs.size(1)]).to(hs.device).to(dtype=torch.long),
            )
            # log_f0_outs, _ = self.f0_predictor(hs.transpose(1, 2), torch.Tensor([hs.size(1)]).to(hs.device).to(dtype=torch.long))
            log_f0_outs = log_f0_outs.transpose(1, 2)
            log_f0_outs = torch.max(
                log_f0_outs, torch.zeros_like(log_f0_outs).to(log_f0_outs)
            )
            hs = hs + self.lf0_mapping(log_f0_outs)

        h_masks = None  # self._source_mask(feats_lengths)
        # forward decoder
        if self.discrete_token_layers > 1 and self.sep:
            before_outs = []
            for i in range(self.discrete_token_layers):
                # print(hs.device, h_masks.device, self.decoder[i].device, flush=True)
                zs, _ = self.decoder[i](hs, h_masks)
                before_outs_ = self.linear_projection[i](zs).view(
                    zs.size(0), -1, self.odim
                )
                before_outs.append(before_outs_)
            before_outs = (
                torch.cat(before_outs, dim=2)
                .view(zs.size(0), -1, self.odim)
                .to(ds.device)
            )
        else:
            zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
            # (B, T_feats, odim), (B, T_feats, 1), (B, T_feats, 1)
            before_outs = self.linear_projection(zs).view(
                zs.size(0), -1, self.odim
            )  # (B, T_feats * layers, nclusters)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        if self.discrete_postnet_layers != 0:
            # print(after_outs.shape, flush=True)
            pred_token = torch.argmax(after_outs, dim=2)
            # print(pred_token.shape, flush=True)
            token_emb = self.token_emb(pred_token)
            # print(token_emb.shape, flush=True)
            dpos_masks = None
            after_outs, _ = self.discrete_postnet(token_emb, dpos_masks)
            after_outs = self.dpos_linear_projection(after_outs)
            # print(after_outs.shape, flush=True)

        if self.use_discrete_token:
            # after_outs input as [B, T, V]

            token_prob = torch.sigmoid(after_outs)
            token_prob = token_prob / token_prob.sum(dim=-1, keepdim=True)
            # Case 1. sample tokens from distribution
            # V = token_prob.size(2)
            # sample_token = torch.multinomial(token_prob.view(-1, V), 1, replacement=True)
            # sample_token = sample_token.view(1, -1, 1) # [B=1, T, S]
            # Case 2. sample
            logits = after_outs
            after_outs = torch.argmax(after_outs, dim=2).unsqueeze(2)
            # if self.codec_codebook > 0:
            #     shift = torch.arange(self.codec_codebook).view(1, 1, -1) * self.odim
            #     after_outs = after_outs.view(1, -1, self.codec_codebook) + shift
            #     after_outs = after_outs.flatten(start_dim=1)
            token = after_outs[0]
            f0 = log_f0_outs[0]
            if use_teacher_forcing:
                layer = self.discrete_token_layers
                token = discrete_token.unsqueeze(-1)
                f0 = pitch
                token_len = len(token) // layer
                if len(f0) > len(token):
                    f0 = f0[:token_len]
                else:
                    f0 = F.pad(f0, (0, 0, 0, token_len - len(f0)), value=0)
            return dict(
                feat_gen=token,
                prob=None,
                att_w=None,
                pitch=f0.squeeze(-1),
                logits=logits,
            )  # outs, probs, att_ws, pitch_outs, logits
        else:
            return dict(
                feat_gen=after_outs[0], prob=None, att_w=None
            )  # outs, probs, att_ws

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            if self.discrete_token_layers > 1 and self.sep:
                for i in range(self.discrete_token_layers):
                    self.decoder[i].embed[-1].alpha.data = torch.tensor(init_dec_alpha)
            else:
                self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
