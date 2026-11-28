import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.s2t.espnet_model import ESPnetS2TModel


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder])
@pytest.mark.parametrize("decoder_arch", [TransformerDecoder])
def test_espnet_model(encoder_arch, decoder_arch):
    token_list = [
        "<blank>",
        "<unk>",
        "<na>",
        "<nospeech>",
        "<en>",
        "<asr>",
        "<st_en>",
        "<notimestamps>",
        "<0.00>",
        "<30.00>",
        "a",
        "i",
        "<sos>",
        "<eos>",
        "<sop>",
    ]
    vocab_size = len(token_list)
    enc_out = 4
    encoder = encoder_arch(
        20, output_size=enc_out, linear_units=4, num_blocks=2, use_flash_attn=False
    )
    decoder = decoder_arch(
        vocab_size, enc_out, linear_units=4, num_blocks=2, use_flash_attn=False
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    model = ESPnetS2TModel(
        vocab_size,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
    )

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_lengths=torch.tensor([4, 3], dtype=torch.long),
        text_prev=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_prev_lengths=torch.tensor([4, 3], dtype=torch.long),
        text_ctc=torch.randint(2, 4, [2, 4], dtype=torch.long),
        text_ctc_lengths=torch.tensor([4, 3], dtype=torch.long),
    )
    loss, *_ = model(**inputs)
    loss.backward()


@pytest.mark.parametrize("encoder_arch", [TransformerEncoder])
@pytest.mark.parametrize("decoder_arch", [TransformerDecoder])
def test_espnet_model_forced_align(encoder_arch, decoder_arch):
    # Run only if torchaudio is available
    pytest.importorskip("torchaudio")
    token_list = [
        "<blank>",
        "<unk>",
        "<na>",
        "<nospeech>",
        "<en>",
        "<asr>",
        "<st_en>",
        "<notimestamps>",
        "<0.00>",
        "<30.00>",
        "a",
        "i",
        "<sos>",
        "<eos>",
        "<sop>",
    ]
    vocab_size = len(token_list)
    enc_out = 4
    encoder = encoder_arch(
        20, output_size=enc_out, linear_units=4, num_blocks=2, use_flash_attn=False
    )
    decoder = decoder_arch(
        vocab_size=vocab_size,
        encoder_output_size=enc_out,
        linear_units=4,
        num_blocks=2,
        use_flash_attn=False,
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)
    model = ESPnetS2TModel(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
    )
    T = 100
    speech = torch.randn(1, T, 20, requires_grad=False)
    speech_lengths = torch.tensor([T], dtype=torch.long)
    L = 5
    text = torch.randint(low=1, high=4, size=(1, L), dtype=torch.long)
    text_lengths = torch.tensor([L], dtype=torch.long)

    aligns, scores = model.forced_align(
        speech=speech,
        speech_lengths=speech_lengths,
        text=text,
        text_lengths=text_lengths,
    )

    assert aligns.shape == (1, T // 4 - 1)
    assert scores.shape == (1, T // 4 - 1)


def test_validation_cer_scores_the_target_after_a_previous_text():
    # With a previous text, the decoder target starts with ignore_id over
    # "<sop> prev". A perfect hypothesis must score CER 0, not 1.
    token_list = ["<blank>", "<unk>", "<na>", "a", "i", "▁", "<sos>", "<eos>", "<sop>"]
    encoder = TransformerEncoder(
        20, output_size=4, linear_units=4, num_blocks=1, use_flash_attn=False
    )
    decoder = TransformerDecoder(
        len(token_list), 4, linear_units=4, num_blocks=1, use_flash_attn=False
    )
    model = ESPnetS2TModel(
        len(token_list),
        token_list=token_list,
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=CTC(odim=len(token_list), encoder_output_size=4),
        sym_na="<na>",
        sym_space="▁",
    )
    ignore = model.ignore_id
    eos = model.eos
    # row 0: prev of 2 tokens -> 3 ignored positions; row 1: no prev
    ys_out_pad = torch.tensor(
        [
            [ignore, ignore, ignore, 3, 5, 4, eos],
            [3, 4, eos, ignore, ignore, ignore, ignore],
        ]
    )
    perfect = ys_out_pad.clamp(min=0)

    hats, refs = model._scored_positions(perfect, ys_out_pad)
    cer, wer = model.error_calculator(hats, refs)

    assert refs.tolist() == [[3, 5, 4, eos], [3, 4, eos, ignore]]
    assert cer == 0.0 and wer == 0.0
    # What the unfiltered tensors give: the first row counted as empty.
    cer_before, _ = model.error_calculator(perfect, ys_out_pad)
    assert cer_before > 0.0
