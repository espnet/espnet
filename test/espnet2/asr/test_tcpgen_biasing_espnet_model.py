import string
from pathlib import Path

import pytest
import sentencepiece as spm
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.tcpgen_biasing_espnet_model import ESPnetTCPGenBiasingASRModel
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer


@pytest.fixture
def spm_srcs(tmp_path: Path):
    input_text = tmp_path / "text"
    vocabsize = len(string.ascii_letters) + 4
    model_prefix = tmp_path / "model"
    model = str(model_prefix) + ".model"
    input_sentence_size = 100000

    with input_text.open("w") as f:
        f.write(string.ascii_letters + "\n")

    spm.SentencePieceTrainer.Train(
        f"--input={input_text} "
        f"--vocab_size={vocabsize} "
        f"--model_prefix={model_prefix} "
        f"--input_sentence_size={input_sentence_size} "
        f"--treat_whitespace_as_suffix=true"
    )
    sp = spm.SentencePieceProcessor()
    sp.load(model)

    with input_text.open("r") as f:
        vocabs = {"<unk>", "▁"}
        for line in f:
            tokens = sp.DecodePieces(list(line.strip()))
        vocabs |= set(tokens)
    return model, vocabs


def pseudo_loss(joint_out, target, t_len, u_len, reduction="", blank=0, gather=True):
    return joint_out.sum() + target.sum()


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("encoder_arch", [ConformerEncoder])
@pytest.mark.parametrize("decoder_arch", [TransducerDecoder])
@pytest.mark.parametrize("multi_blank_durations", [[]])
def test_tcpgen_biasing_espnet_model(
    encoder_arch,
    decoder_arch,
    multi_blank_durations,
    spm_srcs,
    tmp_path: Path,
):
    # Multi-Blank Transducer only supports GPU
    if len(multi_blank_durations) > 0 and not torch.cuda.is_available():
        return
    device = "cuda" if len(multi_blank_durations) > 0 else "cpu"
    device = torch.device(device)

    spm_model, vocabs = spm_srcs
    vocab_size = 5
    enc_out = 4
    encoder = encoder_arch(20, output_size=enc_out, linear_units=4, num_blocks=2)
    decoder = TransducerDecoder(vocab_size, hidden_size=4)
    joint_network = JointNetwork(
        vocab_size,
        encoder_size=enc_out,
        decoder_size=4,
        biasing=True,
        deepbiasing=True,
        biasingsize=2,
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=enc_out)

    biasinglist = tmp_path / "dummy.txt"
    with biasinglist.open("w") as f:
        f.write("A\n")
        f.write("B\n")

    model = ESPnetTCPGenBiasingASRModel(
        vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=joint_network,
        aux_ctc=None,
        transducer_multi_blank_durations=multi_blank_durations,
        biasing=True,
        biasingsche=-1,
        battndim=2,
        deepbiasing=True,
        biasingGNN="gcn1",
        biasinglist=biasinglist,
        bmaxlen=1,
        bdrop=0.0,
        bpemodel=spm_model,
    ).to(device)
    # Have to do this because warp_rnnt only work with CUDA
    model.criterion_transducer_logprob = pseudo_loss

    inputs = dict(
        speech=torch.randn(2, 10, 20, requires_grad=True).to(device),
        speech_lengths=torch.tensor([10, 8], dtype=torch.long).to(device),
        text=torch.randint(2, 4, [2, 4], dtype=torch.long).to(device),
        text_lengths=torch.tensor([4, 3], dtype=torch.long).to(device),
    )
    loss, *_ = model(**inputs)
    loss.backward()
