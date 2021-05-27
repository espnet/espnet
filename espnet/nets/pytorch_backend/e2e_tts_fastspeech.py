# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related modules."""

import logging

import torch
import torch.nn.functional as F

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import (
    DurationCalculator,  # noqa: H301
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
    DurationPredictorLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args


class FeedForwardTransformerLoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.

        """
        super(FeedForwardTransformerLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(self, after_outs, before_outs, d_outs, ys, ds, ilens, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = (
                after_outs.masked_select(out_masks) if after_outs is not None else None
            )
            ys = ys.masked_select(out_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )

        return l1_loss, duration_loss


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.

    This is a module of FastSpeech,
    feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_,
    which does not require any auto-regressive
    processing during inference,
    resulting in fast decoding compared with auto-regressive Transformer.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("feed-forward transformer model setting")
        # network structure related
        group.add_argument(
            "--adim",
            default=384,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        group.add_argument(
            "--elayers", default=6, type=int, help="Number of encoder layers"
        )
        group.add_argument(
            "--eunits", default=1536, type=int, help="Number of encoder hidden units"
        )
        group.add_argument(
            "--dlayers", default=6, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=1536, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--positionwise-layer-type",
            default="linear",
            type=str,
            choices=["linear", "conv1d", "conv1d-linear"],
            help="Positionwise layer type.",
        )
        group.add_argument(
            "--positionwise-conv-kernel-size",
            default=3,
            type=int,
            help="Kernel size of positionwise conv1d layer",
        )
        group.add_argument(
            "--postnet-layers", default=0, type=int, help="Number of postnet layers"
        )
        group.add_argument(
            "--postnet-chans", default=256, type=int, help="Number of postnet channels"
        )
        group.add_argument(
            "--postnet-filts", default=5, type=int, help="Filter size of postnet"
        )
        group.add_argument(
            "--use-batch-norm",
            default=True,
            type=strtobool,
            help="Whether to use batch normalization",
        )
        group.add_argument(
            "--use-scaled-pos-enc",
            default=True,
            type=strtobool,
            help="Use trainable scaled positional encoding "
            "instead of the fixed scale one",
        )
        group.add_argument(
            "--encoder-normalize-before",
            default=False,
            type=strtobool,
            help="Whether to apply layer norm before encoder block",
        )
        group.add_argument(
            "--decoder-normalize-before",
            default=False,
            type=strtobool,
            help="Whether to apply layer norm before decoder block",
        )
        group.add_argument(
            "--encoder-concat-after",
            default=False,
            type=strtobool,
            help="Whether to concatenate attention layer's input and output in encoder",
        )
        group.add_argument(
            "--decoder-concat-after",
            default=False,
            type=strtobool,
            help="Whether to concatenate attention layer's input and output in decoder",
        )
        group.add_argument(
            "--duration-predictor-layers",
            default=2,
            type=int,
            help="Number of layers in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-chans",
            default=384,
            type=int,
            help="Number of channels in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-kernel-size",
            default=3,
            type=int,
            help="Kernel size in duration predictor",
        )
        group.add_argument(
            "--teacher-model",
            default=None,
            type=str,
            nargs="?",
            help="Teacher model file path",
        )
        group.add_argument(
            "--reduction-factor", default=1, type=int, help="Reduction factor"
        )
        group.add_argument(
            "--spk-embed-dim",
            default=None,
            type=int,
            help="Number of speaker embedding dimensions",
        )
        group.add_argument(
            "--spk-embed-integration-type",
            type=str,
            default="add",
            choices=["add", "concat"],
            help="How to integrate speaker embedding",
        )
        # training related
        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="How to initialize transformer parameters",
        )
        group.add_argument(
            "--initial-encoder-alpha",
            type=float,
            default=1.0,
            help="Initial alpha value in encoder's ScaledPositionalEncoding",
        )
        group.add_argument(
            "--initial-decoder-alpha",
            type=float,
            default=1.0,
            help="Initial alpha value in decoder's ScaledPositionalEncoding",
        )
        group.add_argument(
            "--transformer-lr",
            default=1.0,
            type=float,
            help="Initial value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=4000,
            type=int,
            help="Optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-enc-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder except for attention",
        )
        group.add_argument(
            "--transformer-enc-positional-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder positional encoding",
        )
        group.add_argument(
            "--transformer-enc-attn-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder self-attention",
        )
        group.add_argument(
            "--transformer-dec-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer decoder except "
            "for attention and pos encoding",
        )
        group.add_argument(
            "--transformer-dec-positional-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer decoder positional encoding",
        )
        group.add_argument(
            "--transformer-dec-attn-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer decoder self-attention",
        )
        group.add_argument(
            "--transformer-enc-dec-attn-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder-decoder attention",
        )
        group.add_argument(
            "--duration-predictor-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for duration predictor",
        )
        group.add_argument(
            "--postnet-dropout-rate",
            default=0.5,
            type=float,
            help="Dropout rate in postnet",
        )
        group.add_argument(
            "--transfer-encoder-from-teacher",
            default=True,
            type=strtobool,
            help="Whether to transfer teacher's parameters",
        )
        group.add_argument(
            "--transferred-encoder-module",
            default="all",
            type=str,
            choices=["all", "embed"],
            help="Encoder modeules to be trasferred from teacher",
        )
        # loss related
        group.add_argument(
            "--use-masking",
            default=True,
            type=strtobool,
            help="Whether to use masking in calculation of loss",
        )
        group.add_argument(
            "--use-weighted-masking",
            default=False,
            type=strtobool,
            help="Whether to use weighted masking in calculation of loss",
        )
        return parser

    def __init__(self, idim, odim, args=None):
        """Initialize feed-forward Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - elayers (int): Number of encoder layers.
                - eunits (int): Number of encoder hidden units.
                - adim (int): Number of attention transformation dimensions.
                - aheads (int): Number of heads for multi head attention.
                - dlayers (int): Number of decoder layers.
                - dunits (int): Number of decoder hidden units.
                - use_scaled_pos_enc (bool):
                    Whether to use trainable scaled positional encoding.
                - encoder_normalize_before (bool):
                    Whether to perform layer normalization before encoder block.
                - decoder_normalize_before (bool):
                    Whether to perform layer normalization before decoder block.
                - encoder_concat_after (bool): Whether to concatenate attention
                    layer's input and output in encoder.
                - decoder_concat_after (bool): Whether to concatenate attention
                    layer's input and output in decoder.
                - duration_predictor_layers (int): Number of duration predictor layers.
                - duration_predictor_chans (int): Number of duration predictor channels.
                - duration_predictor_kernel_size (int):
                    Kernel size of duration predictor.
                - spk_embed_dim (int): Number of speaker embedding dimensions.
                - spk_embed_integration_type: How to integrate speaker embedding.
                - teacher_model (str): Teacher auto-regressive transformer model path.
                - reduction_factor (int): Reduction factor.
                - transformer_init (float): How to initialize transformer parameters.
                - transformer_lr (float): Initial value of learning rate.
                - transformer_warmup_steps (int): Optimizer warmup steps.
                - transformer_enc_dropout_rate (float):
                    Dropout rate in encoder except attention & positional encoding.
                - transformer_enc_positional_dropout_rate (float):
                    Dropout rate after encoder positional encoding.
                - transformer_enc_attn_dropout_rate (float):
                    Dropout rate in encoder self-attention module.
                - transformer_dec_dropout_rate (float):
                    Dropout rate in decoder except attention & positional encoding.
                - transformer_dec_positional_dropout_rate (float):
                    Dropout rate after decoder positional encoding.
                - transformer_dec_attn_dropout_rate (float):
                    Dropout rate in deocoder self-attention module.
                - transformer_enc_dec_attn_dropout_rate (float):
                    Dropout rate in encoder-deocoder attention module.
                - use_masking (bool):
                    Whether to apply masking for padded part in loss calculation.
                - use_weighted_masking (bool):
                    Whether to apply weighted masking in loss calculation.
                - transfer_encoder_from_teacher:
                    Whether to transfer encoder using teacher encoder parameters.
                - transferred_encoder_module:
                    Encoder module to be initialized using teacher parameters.

        """
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.reduction_factor = args.reduction_factor
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.spk_embed_dim = args.spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = args.spk_embed_integration_type

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=args.adim, padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size,
        )

        # define additional projection for speaker embedding
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, args.adim)
            else:
                self.projection = torch.nn.Linear(
                    args.adim + self.spk_embed_dim, args.adim
                )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=args.adim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=0,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            input_layer=None,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)

        # define postnet
        self.postnet = (
            None
            if args.postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=args.postnet_layers,
                n_chans=args.postnet_chans,
                n_filts=args.postnet_filts,
                use_batch_norm=args.use_batch_norm,
                dropout_rate=args.postnet_dropout_rate,
            )
        )

        # initialize parameters
        self._reset_parameters(
            init_type=args.transformer_init,
            init_enc_alpha=args.initial_encoder_alpha,
            init_dec_alpha=args.initial_decoder_alpha,
        )

        # define teacher model
        if args.teacher_model is not None:
            self.teacher = self._load_teacher_model(args.teacher_model)
        else:
            self.teacher = None

        # define duration calculator
        if self.teacher is not None:
            self.duration_calculator = DurationCalculator(self.teacher)
        else:
            self.duration_calculator = None

        # transfer teacher parameters
        if self.teacher is not None and args.transfer_encoder_from_teacher:
            self._transfer_from_teacher(args.transferred_encoder_module)

        # define criterions
        self.criterion = FeedForwardTransformerLoss(
            use_masking=args.use_masking, use_weighted_masking=args.use_weighted_masking
        )

    def _forward(
        self,
        xs,
        ilens,
        ys=None,
        olens=None,
        spembs=None,
        ds=None,
        is_inference=False,
        alpha=1.0,
    ):
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, alpha)  # (B, Lmax, adim)
        else:
            if ds is None:
                with torch.no_grad():
                    ds = self.duration_calculator(
                        xs, ilens, ys, olens, spembs
                    )  # (B, Tmax)
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            hs = self.length_regulator(hs, ds)  # (B, Lmax, adim)

        # forward decoder
        if olens is not None:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        if is_inference:
            return before_outs, after_outs, d_outs
        else:
            return before_outs, after_outs, ds, d_outs

    def forward(self, xs, ilens, ys, olens, spembs=None, extras=None, *args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).
            extras (Tensor, optional): Batch of precalculated durations (B, Tmax, 1).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        xs = xs[:, : max(ilens)]
        ys = ys[:, : max(olens)]
        if extras is not None:
            extras = extras[:, : max(ilens)].squeeze(-1)

        # forward propagation
        before_outs, after_outs, ds, d_outs = self._forward(
            xs, ilens, ys, olens, spembs=spembs, ds=extras, is_inference=False
        )

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        # calculate loss
        if self.postnet is None:
            l1_loss, duration_loss = self.criterion(
                None, before_outs, d_outs, ys, ds, ilens, olens
            )
        else:
            l1_loss, duration_loss = self.criterion(
                after_outs, before_outs, d_outs, ys, ds, ilens, olens
            )
        loss = l1_loss + duration_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"loss": loss.item()},
        ]

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        self.reporter.report(report_keys)

        return loss

    def calculate_all_attentions(
        self, xs, ilens, ys, olens, spembs=None, extras=None, *args, **kwargs
    ):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).
            extras (Tensor, optional): Batch of precalculated durations (B, Tmax, 1).

        Returns:
            dict: Dict of attention weights and outputs.

        """
        with torch.no_grad():
            # remove unnecessary padded part (for multi-gpus)
            xs = xs[:, : max(ilens)]
            ys = ys[:, : max(olens)]
            if extras is not None:
                extras = extras[:, : max(ilens)].squeeze(-1)

            # forward propagation
            outs = self._forward(
                xs, ilens, ys, olens, spembs=spembs, ds=extras, is_inference=False
            )[1]

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                attn = m.attn.cpu().numpy()
                if "encoder" in name:
                    attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                elif "decoder" in name:
                    if "src" in name:
                        attn = [
                            a[:, :ol, :il]
                            for a, il, ol in zip(attn, ilens.tolist(), olens.tolist())
                        ]
                    elif "self" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, olens.tolist())]
                    else:
                        logging.warning("unknown attention module: " + name)
                else:
                    logging.warning("unknown attention module: " + name)
                att_ws_dict[name] = attn
        att_ws_dict["predicted_fbank"] = [
            m[:l].T for m, l in zip(outs.cpu().numpy(), olens.tolist())
        ]

        return att_ws_dict

    def inference(self, x, inference_args, spemb=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace): Dummy for compatibility.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)
        else:
            spembs = None

        # get option
        alpha = getattr(inference_args, "fastspeech_alpha", 1.0)

        # inference
        _, outs, _ = self._forward(
            xs,
            ilens,
            spembs=spembs,
            is_inference=True,
            alpha=alpha,
        )  # (1, L, odim)

        return outs[0], None, None

    def _integrate_with_spk_embed(self, hs, spembs):
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

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

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).

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

    def _load_teacher_model(self, model_path):
        # get teacher model config
        idim, odim, args = get_model_conf(model_path)

        # assert dimension is the same between teacher and studnet
        assert idim == self.idim
        assert odim == self.odim
        assert args.reduction_factor == self.reduction_factor

        # load teacher model
        from espnet.utils.dynamic_import import dynamic_import

        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args)
        torch_load(model_path, model)

        # freeze teacher model parameters
        for p in model.parameters():
            p.requires_grad = False

        return model

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def _transfer_from_teacher(self, transferred_encoder_module):
        if transferred_encoder_module == "all":
            for (n1, p1), (n2, p2) in zip(
                self.encoder.named_parameters(), self.teacher.encoder.named_parameters()
            ):
                assert n1 == n2, "It seems that encoder structure is different."
                assert p1.shape == p2.shape, "It seems that encoder size is different."
                p1.data.copy_(p2.data)
        elif transferred_encoder_module == "embed":
            student_shape = self.encoder.embed[0].weight.data.shape
            teacher_shape = self.teacher.encoder.embed[0].weight.data.shape
            assert (
                student_shape == teacher_shape
            ), "It seems that embed dimension is different."
            self.encoder.embed[0].weight.data.copy_(
                self.teacher.encoder.embed[0].weight.data
            )
        else:
            raise NotImplementedError("Support only all or embed.")

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        # Lazy import to avoid chainer dependency
        from espnet.nets.pytorch_backend.e2e_tts_transformer import TTSPlot

        return TTSPlot

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training.

        keys should match what `chainer.reporter` reports.
        If you add the key `loss`,
        the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss`
        and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "duration_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]

        return plot_keys
