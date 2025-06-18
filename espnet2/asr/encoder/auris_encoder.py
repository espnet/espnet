import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from espnet2.asr.encoder.beats_encoder import BeatsEncoder, BeatsConfig
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, roll_tensor
from espnet2.speechlm.tokenizer.beats_utils import (
    beats_frontend,
    forward_padding_mask_conv,
)
from contextlib import contextmanager

if torch.__version__ >= "1.6.0":
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class AurisEncoder(BeatsEncoder):
    """changes"""

    def __init__(
        self,
        input_size: int,
        beats_ckpt_path: str = None,
        max_layer: int = None,
        downsampling_rate: int = 1,
        adapter_config: str = "",
        use_weighted_representation: bool = False,
        beats_config: Optional[Dict] = None,
        specaug_config: Optional[Dict] = None,
        add_positional_information: bool = False,
        max_positions: Optional[int] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        roll_augment: bool = False,
        roll_interval: int = 1600,
        is_pretraining: Optional[bool] = False,
    ):
        super().__init__(
            input_size,
            beats_ckpt_path,
            max_layer,
            downsampling_rate,
            adapter_config,
            use_weighted_representation,
            beats_config,
            specaug_config,
            add_positional_information,
            max_positions,
            fbank_mean,
            fbank_std,
            roll_augment,
            roll_interval,
            is_pretraining,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))

    def initialize(self):
        super().initialize()
        if hasattr(self, "cls_token"):
            nn.init.normal_(self.cls_token, std=0.02)
        self.reload_pretrained_parameters()

    def _mask_sequence(self, x, padding_mask, noise=None, mask_ratio=None):
        """Masks the input embedding sequence x for MLM style training.
        Needs self.mask_ratio to be set.

        Args:
            x: [N, L, D], sequence of embeddings.
            padding_mask: [N, L], padding mask for x seq.
                True means padded.
            noise: [N, L], noise for shuffling.
                Generated if not provided.
            mask_ratio: float, ratio of masked positions.
        Returns:
            x_unmasked: [N, l, D], only unmasked portion of
                the input sequence is returned.
            padding_mask: [N, l], portion of padding mask
                corresponding to x_unmasked. True means padded.
            ids_restore: [N, L], restore ids for unshuffling.
                ids_restore[b,j]  = position of x_unmasked[b,j] in x[b].
                No guarantees for masked positions.
            kept: [N, L], binary mask for the unmasked(kept) positions.
                True if the position is kept. Useful for loss computation.
        """
        N, L, D = x.shape  # batch, length, dim

        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        mask_ratio = max(0.51, mask_ratio)

        seq_lengths = (~padding_mask).sum(-1)
        len_keep = (seq_lengths * (1 - mask_ratio)).round().to(dtype=torch.long)
        max_len_kept = len_keep.max()

        if noise is not None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]]
            noise[padding_mask] = float("inf")
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :max_len_kept]

        # make new masks
        padding_mask = make_pad_mask(lengths=len_keep, traceable=False).to(x.device)
        kept = torch.cat(
            [
                ~padding_mask,
                torch.zeros([N, L - max_len_kept], device=x.device, dtype=torch.bool),
            ],
            dim=1,
        )

        # sort only kept indices for maintaining same order x
        ids_keep_sorted = ids_keep.clone()
        ids_keep_sorted = torch.where(
            padding_mask,
            torch.tensor(L - 1, dtype=torch.long, device=x.device),
            ids_keep_sorted,
        )  # introduce L-1 for sorting only important elements
        ids_keep_sorted = ids_keep_sorted.sort(dim=1)[0]
        ids_keep_sorted = torch.where(
            padding_mask, ids_keep, ids_keep_sorted
        )  # handle L-1

        ids_shuffle = torch.cat([ids_keep_sorted, ids_shuffle[:, max_len_kept:]], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_unmasked = torch.gather(
            x, dim=1, index=ids_keep_sorted.unsqueeze(-1).repeat(1, 1, D)
        )

        # unshuffle the loss mask
        kept = torch.gather(kept, dim=1, index=ids_restore)
        return x_unmasked, padding_mask, ids_restore, kept

    def mask_sequence(self, x, padding_mask):
        N, L, D = x.shape
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]]
        noise[padding_mask] = float("inf")

        # batch_mask_ratio = np.random.uniform(0.55, 0.95)
        batch_mask_ratio = None

        x_unmasked1, padding_mask1, ids_restore1, kept1 = self._mask_sequence(
            x, padding_mask, noise=noise, mask_ratio=batch_mask_ratio
        )

        noise = 1 - noise
        x_unmasked2, padding_mask2, ids_restore2, kept2 = self._mask_sequence(
            x, padding_mask, noise=noise, mask_ratio=batch_mask_ratio
        )

        # Make shape compatible for concatenation
        maxlen = max(x_unmasked1.shape[1], x_unmasked2.shape[1])
        if x_unmasked1.shape[1] < maxlen:
            l1 = maxlen - x_unmasked1.shape[1]
            x_unmasked1 = torch.cat(
                [
                    x_unmasked1,
                    torch.zeros(
                        N, l1, D, dtype=x_unmasked1.dtype, device=x_unmasked1.device
                    ),
                ],
                dim=1,
            )
            padding_mask1 = torch.cat(
                [
                    padding_mask1,
                    torch.ones(
                        N, l1, dtype=padding_mask1.dtype, device=padding_mask1.device
                    ),
                ],
                dim=1,
            )
            kept1 = torch.cat(
                [
                    kept1,
                    torch.zeros(
                        N,
                        maxlen - kept1.shape[1],
                        dtype=kept1.dtype,
                        device=kept1.device,
                    ),
                ],
                dim=1,
            )
        elif x_unmasked2.shape[1] < maxlen:
            l2 = maxlen - x_unmasked2.shape[1]
            x_unmasked2 = torch.cat(
                [
                    x_unmasked2,
                    torch.zeros(
                        N, l2, D, dtype=x_unmasked2.dtype, device=x_unmasked2.device
                    ),
                ],
                dim=1,
            )
            padding_mask2 = torch.cat(
                [
                    padding_mask2,
                    torch.ones(
                        N, l2, dtype=padding_mask2.dtype, device=padding_mask2.device
                    ),
                ],
                dim=1,
            )

        # Combine into one batch
        x_unmasked = torch.cat([x_unmasked1, x_unmasked2], dim=0)
        padding_mask = torch.cat([padding_mask1, padding_mask2], dim=0)
        ids_restore = torch.cat([ids_restore1, ids_restore2], dim=0)
        kept = torch.cat([kept1, kept2], dim=0)

        # prefix CLS
        x_unmasked = torch.cat(
            [self.cls_token.expand(2 * N, 1, D), x_unmasked],
            dim=1,
        )
        padding_mask = torch.cat(
            [
                torch.zeros_like(padding_mask[:, :1]),
                padding_mask,
            ],
            dim=1,
        )
        ids_restore = torch.cat(
            [torch.zeros_like(ids_restore[:, :1]), ids_restore + 1],
            dim=1,
        )
        kept = torch.cat(
            [torch.ones_like(kept[:, :1]), kept],
            dim=1,
        )
        return x_unmasked, padding_mask, ids_restore, kept

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        max_layer: Optional[int] = None,
        skip_fbank_extraction: bool = False,
    ):
        """Extract features from raw audio.
        source: (B,T,D). for waveform input D=1, for features D=feature_dim
        padding_mask: (B,T). If True then pad the element.
        """
        if (
            not skip_fbank_extraction
            and self.min_input_length_at_16khz
            and source.size(1) < self.min_input_length_at_16khz
        ):
            # Only executed for raw waveform input
            logging.warning(
                f"Input shape: {source.shape}. This is less than"
                f" the minimum size of {self.min_input_length_at_16khz}."
            )
            # repeat the input to make it at least min_length
            repeat_factor = self.min_input_length_at_16khz // source.size(1) + 1
            source = torch.cat([source] * repeat_factor, dim=1)
            padding_mask = torch.cat([padding_mask] * repeat_factor, dim=1)

        with autocast(False):
            fbank = (
                (source - self.fbank_mean) / (2 * self.fbank_std)
                if skip_fbank_extraction
                else beats_frontend(
                    source.squeeze(-1),
                    fbank_mean=self.fbank_mean,
                    fbank_std=self.fbank_std,
                )
            )

            if self.specaug is not None and self.training:
                fbank = self.specaug(fbank)[0]

        if padding_mask is not None and not skip_fbank_extraction:
            # padding_mask = self.forward_padding_mask(fbank, padding_mask)
            padding_mask = forward_padding_mask_conv(
                padding_mask=padding_mask, n_dim=0, conv_module=self.raw2fbank_pad
            )

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)

        if padding_mask is not None:
            # features is BTC
            padding_mask = forward_padding_mask_conv(
                padding_mask=padding_mask,
                n_dim=fbank.shape[-1],
                conv_module=self.patch_embedding_pad,
            )

        patch_padding_mask = None
        restore_ids = None
        kept_mask = None
        if self.is_pretraining:
            assert (
                max_layer is None
            ), "During pretraining max_layer should be set to None!"
            patch_padding_mask = padding_mask.clone()
            # kept_mask: 1 - kept, 0 - removed, corresponding to features
            # features, padding_mask will be shortened to only keep the kept positions
            features, padding_mask, restore_ids, kept_mask = self.mask_sequence(
                features, padding_mask
            )
        else:
            # Add CLS
            features = torch.cat(
                [
                    self.cls_token.expand(features.shape[0], -1, -1),
                    features,
                ],
                dim=1,
            )
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        features.shape[0],
                        1,
                        device=features.device,
                        dtype=padding_mask.dtype,
                    ),
                    padding_mask,
                ],
                dim=1,
            )

        features = self.layer_norm(features)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        features, layer_results = self.encoder(
            features, padding_mask=padding_mask, layer=max_layer
        )

        if max_layer is not None:
            features = layer_results[max_layer][0].transpose(
                0, 1
            )  # use the output from the max_layer

        if self.use_weighted_representation:
            repr_layer_weights = nn.functional.softmax(self.layer_weights, dim=-2)
            assert (
                max_layer is not None
            ), "max_layer must not be None when using weighted representations."
            features = (
                torch.stack(
                    [
                        layer_result_i.transpose(0, 1)
                        for layer_result_i, _ in layer_results[: max_layer + 1]
                    ],
                    dim=-2,
                )
                * repr_layer_weights
            )
            features = features.sum(dim=-2)  # BTC

        if self.downsample_conv is not None:
            features = self.downsample_conv(features.transpose(1, 2)).transpose(
                1, 2
            )  # BTC
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.conformer_adapter:
            # to handle incompatibility btw torch & huggingface
            conformer_attn_mask = ~padding_mask
            # run through conformer
            features = self.conformer_adapter(
                features,
                attention_mask=conformer_attn_mask,
            ).last_hidden_state

        if self.cross_embed_positions is not None:
            features = features + self.cross_embed_positions(features)

        return features, padding_mask, restore_ids, kept_mask, patch_padding_mask
