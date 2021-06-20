# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TTS-Transformer related modules."""

from typing import Dict
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.e2e_tts_transformer import GuidedMultiHeadAttentionLoss
from espnet.nets.pytorch_backend.e2e_tts_transformer import TransformerLoss
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gst.style_encoder import StyleEncoder


class Transformer(AbsTTS):
    """TTS-Transformer module.

    This is a module of text-to-speech Transformer described in `Neural Speech Synthesis
    with Transformer Network`_, which convert the sequence of tokens into the sequence
    of Mel-filterbanks.

    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        embed_dim (int, optional): Dimension of character embedding.
        eprenet_conv_layers (int, optional):
            Number of encoder prenet convolution layers.
        eprenet_conv_chans (int, optional):
            Number of encoder prenet convolution channels.
        eprenet_conv_filts (int, optional):
            Filter size of encoder prenet convolution.
        dprenet_layers (int, optional): Number of decoder prenet layers.
        dprenet_units (int, optional): Number of decoder prenet hidden units.
        elayers (int, optional): Number of encoder layers.
        eunits (int, optional): Number of encoder hidden units.
        adim (int, optional): Number of attention transformation dimensions.
        aheads (int, optional): Number of heads for multi head attention.
        dlayers (int, optional): Number of decoder layers.
        dunits (int, optional): Number of decoder hidden units.
        postnet_layers (int, optional): Number of postnet layers.
        postnet_chans (int, optional): Number of postnet channels.
        postnet_filts (int, optional): Filter size of postnet.
        use_scaled_pos_enc (bool, optional):
            Whether to use trainable scaled positional encoding.
        use_batch_norm (bool, optional):
            Whether to use batch normalization in encoder prenet.
        encoder_normalize_before (bool, optional):
            Whether to perform layer normalization before encoder block.
        decoder_normalize_before (bool, optional):
            Whether to perform layer normalization before decoder block.
        encoder_concat_after (bool, optional): Whether to concatenate attention
            layer's input and output in encoder.
        decoder_concat_after (bool, optional): Whether to concatenate attention
            layer's input and output in decoder.
        positionwise_layer_type (str, optional):
            Position-wise operation type.
        positionwise_conv_kernel_size (int, optional):
            Kernel size in position wise conv 1d.
        reduction_factor (int, optional): Reduction factor.
        spk_embed_dim (int, optional): Number of speaker embedding dimenstions.
        spk_embed_integration_type (str, optional): How to integrate speaker embedding.
        use_gst (str, optional): Whether to use global style token.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        gst_conv_layers (int, optional): The number of conv layers in GST.
        gst_conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in GST.
        gst_conv_kernel_size (int, optional): Kernal size of conv layers in GST.
        gst_conv_stride (int, optional): Stride size of conv layers in GST.
        gst_gru_layers (int, optional): The number of GRU layers in GST.
        gst_gru_units (int, optional): The number of GRU units in GST.
        transformer_lr (float, optional): Initial value of learning rate.
        transformer_warmup_steps (int, optional): Optimizer warmup steps.
        transformer_enc_dropout_rate (float, optional):
            Dropout rate in encoder except attention and positional encoding.
        transformer_enc_positional_dropout_rate (float, optional):
            Dropout rate after encoder positional encoding.
        transformer_enc_attn_dropout_rate (float, optional):
            Dropout rate in encoder self-attention module.
        transformer_dec_dropout_rate (float, optional):
            Dropout rate in decoder except attention & positional encoding.
        transformer_dec_positional_dropout_rate (float, optional):
            Dropout rate after decoder positional encoding.
        transformer_dec_attn_dropout_rate (float, optional):
            Dropout rate in deocoder self-attention module.
        transformer_enc_dec_attn_dropout_rate (float, optional):
            Dropout rate in encoder-deocoder attention module.
        init_type (str, optional):
            How to initialize transformer parameters.
        init_enc_alpha (float, optional):
            Initial value of alpha in scaled pos encoding of the encoder.
        init_dec_alpha (float, optional):
            Initial value of alpha in scaled pos encoding of the decoder.
        eprenet_dropout_rate (float, optional): Dropout rate in encoder prenet.
        dprenet_dropout_rate (float, optional): Dropout rate in decoder prenet.
        postnet_dropout_rate (float, optional): Dropout rate in postnet.
        use_masking (bool, optional):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool, optional):
            Whether to apply weighted masking in loss calculation.
        bce_pos_weight (float, optional): Positive sample weight in bce calculation
            (only for use_masking=true).
        loss_type (str, optional): How to calculate loss.
        use_guided_attn_loss (bool, optional): Whether to use guided attention loss.
        num_heads_applied_guided_attn (int, optional):
            Number of heads in each layer to apply guided attention loss.
        num_layers_applied_guided_attn (int, optional):
            Number of layers to apply guided attention loss.
        modules_applied_guided_attn (Sequence[str], optional):
            List of module names to apply guided attention loss.
        guided_attn_loss_sigma (float, optional) Sigma in guided attention loss.
        guided_attn_loss_lambda (float, optional): Lambda in guided attention loss.

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        embed_dim: int = 512,
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        dprenet_layers: int = 2,
        dprenet_units: int = 256,
        elayers: int = 6,
        eunits: int = 1024,
        adim: int = 512,
        aheads: int = 4,
        dlayers: int = 6,
        dunits: int = 1024,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        reduction_factor: int = 1,
        spk_embed_dim: int = None,
        spk_embed_integration_type: str = "add",
        use_gst: bool = False,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        transformer_enc_dec_attn_dropout_rate: float = 0.1,
        eprenet_dropout_rate: float = 0.5,
        dprenet_dropout_rate: float = 0.5,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        loss_type: str = "L1",
        use_guided_attn_loss: bool = True,
        num_heads_applied_guided_attn: int = 2,
        num_layers_applied_guided_attn: int = 2,
        modules_applied_guided_attn: Sequence[str] = ("encoder-decoder"),
        guided_attn_loss_sigma: float = 0.4,
        guided_attn_loss_lambda: float = 1.0,
    ):
        """Initialize Transformer module."""
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        self.reduction_factor = reduction_factor
        self.use_gst = use_gst
        self.use_guided_attn_loss = use_guided_attn_loss
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            if num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = elayers
            else:
                self.num_layers_applied_guided_attn = num_layers_applied_guided_attn
            if num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = aheads
            else:
                self.num_heads_applied_guided_attn = num_heads_applied_guided_attn
            self.modules_applied_guided_attn = modules_applied_guided_attn
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, adim),
            )
        else:
            encoder_input_layer = torch.nn.Embedding(
                num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
            )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
            )

        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define transformer decoder
        if dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=odim,
                    n_layers=dprenet_layers,
                    n_units=dprenet_units,
                    dropout_rate=dprenet_dropout_rate,
                ),
                torch.nn.Linear(dprenet_units, adim),
            )
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(
            odim=odim,  # odim is needed when no prenet is used
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.prob_out = torch.nn.Linear(adim, reduction_factor)

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

        # define loss function
        self.criterion = TransformerLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
            )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = speech
        olens = speech_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate transformer outputs
        after_outs, before_outs, logits = self._forward(xs, ilens, ys, olens, spembs)

        # modifiy mod part of groundtruth
        olens_in = olens
        if self.reduction_factor > 1:
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # caluculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs, before_outs, logits, ys, labels, olens
        )
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        stats = dict(
            l1_loss=l1_loss.item(),
            l2_loss=l2_loss.item(),
            bce_loss=bce_loss.item(),
        )

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.encoder.encoders)))
                ):
                    att_ws += [
                        self.encoder.encoders[layer_idx].self_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_in, T_in)
                enc_attn_loss = self.attn_criterion(att_ws, ilens, ilens)
                loss = loss + enc_attn_loss
                stats.update(enc_attn_loss=enc_attn_loss.item())
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].self_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_out)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
                stats.update(dec_attn_loss=dec_attn_loss.item())
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].src_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_in)
                enc_dec_attn_loss = self.attn_criterion(att_ws, ilens, olens_in)
                loss = loss + enc_dec_attn_loss
                stats.update(enc_dec_attn_loss=enc_dec_attn_loss.item())

        stats.update(loss=loss.item())

        # report extra information
        if self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        spembs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, h_masks = self.encoder(xs, x_masks)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1 :: self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, h_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs, before_outs, logits

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        spembs: torch.Tensor = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            spembs (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            threshold (float, optional): Threshold in inference.
            minlenratio (float, optional): Minimum length ratio in inference.
            maxlenratio (float, optional): Maximum length ratio in inference.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """
        x = text
        y = speech
        spemb = spembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            # get teacher forcing outputs
            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, *_ = self._forward(xs, ilens, ys, olens, spembs)

            # get attention weights
            att_ws = []
            for i in range(len(self.decoder.decoders)):
                att_ws += [self.decoder.decoders[i].src_attn.attn]
            att_ws = torch.stack(att_ws, dim=1)  # (B, L, H, T_out, T_in)

            return outs[0], None, att_ws[0]

        # forward encoder
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)

        # integrate GST
        if self.use_gst:
            style_embs = self.gst(y.unsqueeze(0))
            hs = hs + style_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            spembs = spemb.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, spembs)

        # set limits of length
        maxlen = int(hs.size(1) * maxlenratio / self.reduction_factor)
        minlen = int(hs.size(1) * minlenratio / self.reduction_factor)

        # initialize
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        z_cache = self.decoder.init_state(x)
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
            z, z_cache = self.decoder.forward_one_step(
                ys, y_masks, hs, cache=z_cache
            )  # (B, adim)
            outs += [
                self.feat_out(z).view(self.reduction_factor, self.odim)
            ]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat(
                (ys, outs[-1][-1].view(1, 1, self.odim)), dim=1
            )  # (1, idx + 1, odim)

            # get attention weights
            att_ws_ = []
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]  # [(#heads, 1, T),...]
            if idx == 1:
                att_ws = att_ws_
            else:
                # [(#heads, l, T), ...]
                att_ws = [
                    torch.cat([att_w, att_w_], dim=1)
                    for att_w, att_w_ in zip(att_ws, att_ws_)
                ]

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = (
                    torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2)
                )  # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                break

        # concatenate attention weights -> (#layers, #heads, L, T)
        att_ws = torch.stack(att_ws, dim=0)

        return outs, probs, att_ws

    def _add_first_frame_and_remove_last_frame(self, ys: torch.Tensor) -> torch.Tensor:
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

    def _source_mask(self, ilens):
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
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens: torch.Tensor) -> torch.Tensor:
        """Make masks for masked self-attention.

        Args:
            olens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

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
