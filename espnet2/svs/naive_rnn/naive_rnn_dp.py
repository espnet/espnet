# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""NaiveRNN-DP-SVS related modules."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.svs.abs_svs import AbsSVS
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet.nets.pytorch_backend.e2e_tts_fastspeech import (
    FeedForwardTransformerLoss as FastSpeechLoss,
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet


class NaiveRNNDP(AbsSVS):
    """NaiveRNNDP-SVS module.

    This is an implementation of naive RNN with duration prediction
    for singing voice synthesis
    The features are processed directly over time-domain from music score and
    predict the singing voice features
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        midi_dim: int = 129,
        embed_dim: int = 512,
        tempo_dim: int = 500,
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        elayers: int = 3,
        eunits: int = 1024,
        ebidirectional: bool = True,
        midi_embed_integration_type: str = "add",
        dlayers: int = 3,
        dunits: int = 1024,
        dbidirectional: bool = True,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        use_batch_norm: bool = True,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        reduction_factor: int = 1,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        eprenet_dropout_rate: float = 0.5,
        edropout_rate: float = 0.1,
        ddropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        """Initialize NaiveRNN module.

        Args: TODO(Yuning)
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.tempo_dim = tempo_dim
        self.eunits = eunits
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor

        self.midi_embed_integration_type = midi_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            self.encoder_input_layer = torch.nn.Sequential(
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
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
            self.midi_encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=midi_dim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
            self.tempo_encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=midi_dim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
        else:
            self.encoder_input_layer = torch.nn.Embedding(
                num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
            )
            self.midi_encoder_input_layer = torch.nn.Embedding(
                num_embeddings=midi_dim,
                embedding_dim=eunits,
                padding_idx=self.padding_idx,
            )
            self.tempo_encoder_input_layer = torch.nn.Embedding(
                num_embeddings=tempo_dim,
                embedding_dim=eunits,
                padding_idx=self.padding_idx,
            )

        self.encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.midi_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        self.tempo_encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            # proj_size=eunits,
        )

        dim_direction = 2 if ebidirectional is True else 1
        if self.midi_embed_integration_type == "add":
            self.midi_projection = torch.nn.Linear(
                eunits * dim_direction, eunits * dim_direction
            )
        else:
            self.midi_projection = torch.nn.linear(
                3 * eunits * dim_direction, eunits * dim_direction
            )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=eunits * dim_direction,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        self.decoder = torch.nn.LSTM(
            input_size=eunits * dim_direction,
            hidden_size=dunits,
            num_layers=dlayers,
            batch_first=True,
            dropout=ddropout_rate,
            bidirectional=dbidirectional,
            # proj_size=dunits,
        )

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, dunits * dim_direction)
        self.langs = None
        if langs is not None and langs > 1:
            # TODO(Yuning): not encode yet
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, dunits * dim_direction)

        # define projection layer
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(
                    self.spk_embed_dim, dunits * dim_direction
                )
            else:
                self.projection = torch.nn.Linear(
                    dunits * dim_direction + self.spk_embed_dim, dunits * dim_direction
                )

        # define final projection
        self.feat_out = torch.nn.Linear(dunits * dim_direction, odim * reduction_factor)

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
        self.criterion = FastSpeechLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
        )

    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

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
        tempo: Optional[Dict[str, torch.Tensor]] = None,
        tempo_lengths: Optional[Dict[str, torch.Tensor]] = None,
        beat: Optional[Dict[str, torch.Tensor]] = None,
        beat_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            melody_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded melody (B, ).
            tempo (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded tempo (B, Tmax).
            tempo_lengths (Optional[Dict]):  key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded tempo (B, ).
            beat (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded beat (B, Tmax).
            beat_length (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of the lengths of padded beat (B, ).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            pitch_lengths (LongTensor): Batch of the lengths of padded f0 (B, ).
            duration (Optional[Dict]): key is "phn", "syb";
                value (LongTensor): Batch of padded beat (B, Tmax).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).

        GS Fix:
            arguements from forward func. V.S. **batch from espnet_model.py
            label == durations | phone sequence
            melody -> pitch sequence

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """
        label = label["score"]
        midi = melody["score"]
        tempo = beat["score_phn"]
        label_lengths = label_lengths["score"]
        midi_lengths = melody_lengths["score"]
        ds = duration["phn"]

        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        batch_size = feats.size(0)

        label_emb = self.encoder_input_layer(label)  # FIX ME: label Float to Int
        midi_emb = self.midi_encoder_input_layer(midi)
        tempo_emb = self.tempo_encoder_input_layer(tempo)

        label_emb = torch.nn.utils.rnn.pack_padded_sequence(
            label_emb, label_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        midi_emb = torch.nn.utils.rnn.pack_padded_sequence(
            midi_emb, midi_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        tempo_emb = torch.nn.utils.rnn.pack_padded_sequence(
            tempo_emb, midi_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )

        hs_label, (_, _) = self.encoder(label_emb)
        hs_midi, (_, _) = self.midi_encoder(midi_emb)
        hs_tempo, (_, _) = self.tempo_encoder(tempo_emb)

        hs_label, _ = torch.nn.utils.rnn.pad_packed_sequence(hs_label, batch_first=True)
        hs_midi, _ = torch.nn.utils.rnn.pad_packed_sequence(hs_midi, batch_first=True)
        hs_tempo, _ = torch.nn.utils.rnn.pad_packed_sequence(hs_tempo, batch_first=True)

        if self.midi_embed_integration_type == "add":
            hs = hs_label + hs_midi + hs_tempo
            hs = F.leaky_relu(self.midi_projection(hs))
        else:
            hs = torch.cat((hs_label, hs_midi, hs_tempo), dim=-1)
            hs = F.leaky_relu(self.midi_projection(hs))
        # integrate spk & lang embeddings
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
        d_masks = make_pad_mask(label_lengths).to(hs.device)
        d_outs = self.duration_predictor(hs, d_masks)  # (B, T_text)
        hs = self.length_regulator(hs, ds)  # (B, seq_len, eunits)

        hs_emb = torch.nn.utils.rnn.pack_padded_sequence(
            hs, feats_lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )

        zs, (_, _) = self.decoder(hs_emb)
        zs, _ = torch.nn.utils.rnn.pad_packed_sequence(zs, batch_first=True)

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

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
        else:
            ys = feats
            olens = feats_lengths

        # calculate loss values
        ilens = label_lengths
        l1_loss, duration_loss = self.criterion(
            after_outs, before_outs, d_outs, ys, ds, ilens, olens
        )
        loss = l1_loss + duration_loss

        stats = dict(
            loss=loss.item(), l1_loss=l1_loss.item(), duration_loss=duration_loss.item()
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        if flag_IsValid is False:
            # training stage
            return loss, stats, weight
        else:
            # validation stage
            return loss, stats, weight, after_outs[:, : olens.max()], ys, olens

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        tempo: Optional[Dict[str, torch.Tensor]] = None,
        beat: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (Tmax).
            feats (Tensor): Batch of padded target features (Lmax, odim).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (Tmax).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (Tmax).
            tempo (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded tempo (Tmax).
            beat (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded beat (Tmax).
            pitch (FloatTensor): Batch of padded f0 (Tmax).
            duration (Optional[Dict]): key is "phn", "syb";
                value (LongTensor): Batch of padded beat (Tmax).
            spembs (Optional[Tensor]): Batch of speaker embeddings (spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (1).
            lids (Optional[Tensor]): Batch of language IDs (1).

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
        """
        label = label["score"]
        midi = melody["score"]
        tempo = beat["score_phn"]

        label_emb = self.encoder_input_layer(label)  # FIX ME: label Float to Int
        midi_emb = self.midi_encoder_input_layer(midi)
        tempo_emb = self.tempo_encoder_input_layer(tempo)

        hs_label, (_, _) = self.encoder(label_emb)
        hs_midi, (_, _) = self.midi_encoder(midi_emb)
        hs_tempo, (_, _) = self.tempo_encoder(tempo_emb)

        if self.midi_embed_integration_type == "add":
            hs = hs_label + hs_midi + hs_tempo
            hs = F.leaky_relu(self.midi_projection(hs))
        else:
            hs = torch.cat((hs_label, hs_midi, hs_tempo), dim=-1)
            hs = F.leaky_relu(self.midi_projection(hs))
        # integrate spk & lang embeddings
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
        d_masks = None  # make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
        d_outs_int = torch.floor(d_outs + 0.5).to(dtype=torch.long)  # (B, T_text)

        hs = self.length_regulator(hs, d_outs_int)  # (B, T_feats, adim)
        zs, (_, _) = self.decoder(hs)

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs, None, None  # outs, probs, att_ws

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

        return
