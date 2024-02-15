import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.s2st.aux_attention.multihead import MultiHeadAttention
from espnet2.s2st.espnet_model import ESPnetS2STModel
from espnet2.s2st.losses.attention_loss import S2STAttentionLoss
from espnet2.s2st.losses.ctc_loss import S2STCTCLoss
from espnet2.s2st.losses.guided_attention_loss import S2STGuidedAttentionLoss
from espnet2.s2st.losses.tacotron_loss import S2STTacotron2Loss
from espnet2.s2st.synthesizer.discrete_synthesizer import TransformerDiscreteSynthesizer
from espnet2.s2st.synthesizer.translatotron import Translatotron


@pytest.mark.parametrize("encoder", [ConformerEncoder, TransformerEncoder])
def test_espnet_model_discrete_unit(encoder):
    vocab_size = 5
    enc_out = 4
    encoder = encoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
    asr_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    st_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    synthesizer = TransformerDiscreteSynthesizer(
        vocab_size, enc_out, linear_units=4, num_blocks=2
    )
    asr_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)
    st_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    losses = {
        "src_attn": S2STAttentionLoss(vocab_size),
        "tgt_attn": S2STAttentionLoss(vocab_size),
        "synthesis": S2STAttentionLoss(vocab_size),
        "asr_ctc": S2STCTCLoss(),
        "st_ctc": S2STCTCLoss(),
    }

    model = ESPnetS2STModel(
        s2st_type="discrete_unit",
        frontend=None,
        src_normalize=None,
        tgt_normalize=None,
        tgt_feats_extract=None,
        specaug=None,
        encoder=encoder,
        preencoder=None,
        postencoder=None,
        aux_attention=None,
        unit_encoder=None,
        losses=losses,
        asr_decoder=asr_decoder,
        st_decoder=st_decoder,
        synthesizer=synthesizer,
        asr_ctc=asr_ctc,
        st_ctc=st_ctc,
        tgt_vocab_size=vocab_size,
        tgt_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        src_vocab_size=vocab_size,
        src_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        unit_vocab_size=vocab_size,
        unit_token_list=["<blank>", "<unk>", "0", "1", "<eos>"],
    )

    inputs = dict(
        src_speech=torch.randn(2, 10, 20, requires_grad=True),
        src_speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        src_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        src_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        tgt_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_speech=torch.randint(2, 4, [2, 4], dtype=torch.long),
        tgt_speech_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()


@pytest.mark.parametrize("encoder", [ConformerEncoder, TransformerEncoder])
def test_espnet_model_translatotron(encoder):
    vocab_size = 5
    enc_out = 4
    encoder = encoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
    asr_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    st_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    synthesizer = Translatotron(
        enc_out,
        15,
        embed_dim=4,
        adim=4,
        dunits=2,
        dlayers=2,
        prenet_layers=1,
        postnet_layers=1,
        postnet_chans=2,
    )
    asr_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)
    st_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    losses = {
        "src_attn": S2STAttentionLoss(vocab_size),
        "tgt_attn": S2STAttentionLoss(vocab_size),
        "synthesis": S2STTacotron2Loss(),
        "asr_ctc": S2STCTCLoss(),
        "st_ctc": S2STCTCLoss(),
    }

    model = ESPnetS2STModel(
        s2st_type="translatotron",
        frontend=None,
        src_normalize=None,
        tgt_normalize=None,
        tgt_feats_extract=None,
        specaug=None,
        encoder=encoder,
        preencoder=None,
        postencoder=None,
        aux_attention=None,
        unit_encoder=None,
        losses=losses,
        asr_decoder=asr_decoder,
        st_decoder=st_decoder,
        synthesizer=synthesizer,
        asr_ctc=asr_ctc,
        st_ctc=st_ctc,
        tgt_vocab_size=vocab_size,
        tgt_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        src_vocab_size=vocab_size,
        src_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        unit_vocab_size=None,
        unit_token_list=None,
    )

    inputs = dict(
        src_speech=torch.randn(2, 10, 20, requires_grad=True),
        src_speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        src_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        src_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        tgt_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_speech=torch.randn(2, 10, 15, requires_grad=True),
        tgt_speech_lengths=torch.tensor([10, 8], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()


@pytest.mark.parametrize("encoder", [ConformerEncoder, TransformerEncoder])
def test_espnet_model_unity(encoder):
    vocab_size = 5
    enc_out = 4
    speech_encoder = encoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
    unit_encoder = encoder(
        enc_out, output_size=enc_out, linear_units=4, num_blocks=2, input_layer=None
    )
    asr_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    st_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    synthesizer = TransformerDiscreteSynthesizer(
        vocab_size, enc_out, linear_units=4, num_blocks=2
    )
    asr_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)
    st_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    losses = {
        "src_attn": S2STAttentionLoss(vocab_size),
        "tgt_attn": S2STAttentionLoss(vocab_size),
        "synthesis": S2STAttentionLoss(vocab_size),
        "asr_ctc": S2STCTCLoss(),
        "st_ctc": S2STCTCLoss(),
    }

    model = ESPnetS2STModel(
        s2st_type="unity",
        frontend=None,
        src_normalize=None,
        tgt_normalize=None,
        tgt_feats_extract=None,
        specaug=None,
        encoder=speech_encoder,
        preencoder=None,
        postencoder=None,
        aux_attention=None,
        unit_encoder=unit_encoder,
        losses=losses,
        asr_decoder=asr_decoder,
        st_decoder=st_decoder,
        synthesizer=synthesizer,
        asr_ctc=asr_ctc,
        st_ctc=st_ctc,
        tgt_vocab_size=vocab_size,
        tgt_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        src_vocab_size=vocab_size,
        src_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        unit_vocab_size=5,
        unit_token_list=["<blank>", "<unk>", "0", "1", "<eos>"],
    )

    inputs = dict(
        src_speech=torch.randn(2, 10, 20, requires_grad=True),
        src_speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        src_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        src_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        tgt_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_speech=torch.randint(2, 4, [2, 4], dtype=torch.long),
        tgt_speech_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()


@pytest.mark.parametrize("encoder", [ConformerEncoder, TransformerEncoder])
def test_espnet_model_translatotron2(encoder):
    vocab_size = 5
    enc_out = 4
    encoder = encoder(20, output_size=enc_out, linear_units=4, num_blocks=2)
    asr_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    st_decoder = TransformerDecoder(vocab_size, enc_out, linear_units=4, num_blocks=2)
    synthesizer = Translatotron(
        2 * enc_out,
        15,
        embed_dim=4,
        adim=4,
        dunits=2,
        dlayers=2,
        prenet_layers=1,
        postnet_layers=1,
        postnet_chans=2,
    )
    asr_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)
    st_ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)
    aux_attention = MultiHeadAttention(n_feat=enc_out)

    losses = {
        "src_attn": S2STAttentionLoss(vocab_size),
        "tgt_attn": S2STAttentionLoss(vocab_size),
        "synthesis": S2STTacotron2Loss(),
        "asr_ctc": S2STCTCLoss(),
        "st_ctc": S2STCTCLoss(),
    }

    model = ESPnetS2STModel(
        s2st_type="translatotron2",
        frontend=None,
        src_normalize=None,
        tgt_normalize=None,
        tgt_feats_extract=None,
        specaug=None,
        encoder=encoder,
        preencoder=None,
        postencoder=None,
        aux_attention=aux_attention,
        unit_encoder=None,
        losses=losses,
        asr_decoder=asr_decoder,
        st_decoder=st_decoder,
        synthesizer=synthesizer,
        asr_ctc=asr_ctc,
        st_ctc=st_ctc,
        tgt_vocab_size=vocab_size,
        tgt_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        src_vocab_size=vocab_size,
        src_token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        unit_vocab_size=None,
        unit_token_list=None,
    )

    inputs = dict(
        src_speech=torch.randn(2, 10, 20, requires_grad=True),
        src_speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        src_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        src_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        tgt_text_lengths=torch.tensor([4, 3], dtype=torch.long),
        tgt_speech=torch.randn(2, 10, 15, requires_grad=True),
        tgt_speech_lengths=torch.tensor([10, 8], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()
