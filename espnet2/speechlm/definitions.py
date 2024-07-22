#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from dataclasses import dataclass
from typing import List, Tuple


# (1) Modality definition
# a. discrete means the discrete / continuous format in final sequences.
#    e.g., the LLM output embeddings (textemb) is originally read as text
#    then tokenized into BPE, and finally converted into continuous
#    embeddings before being processed by the SpeechLM. So it's continouos
#    This is to determine if a placeholder should be adopted in the spliced
#    sequence in preprocess_fn
# b. data_type: how the original data file is loaded. This is exactly follows
#    the definitions in espent2.train.dataset
# c. For discrete modality, we usually have a modality-specific vocabulary
#    an exception is "spk"
@dataclass
class Modality:
    discrete: bool = (True,)


modalities = {}
# Discrete
modalities["codec"] = Modality()
modalities["ssl"] = Modality()
modalities["text_bpe"] = Modality()
modalities["g2p"] = Modality()
modalities["spk"] = Modality()
modalities["class"] = Modality()
# Continuous
modalities["wav"] = Modality(discrete=False)
modalities["text_emb"] = Modality(discrete=False)
modalities["ssl_feat"] = Modality(discrete=False)

############### END OF MODALITY DEFINITION ###############


# (2) Task Definition
@dataclass
class SpeechLMTask:
    encoder_entries: List[Tuple[str, str, str]]
    decoder_entries: List[Tuple[str, str, str]]
    target_entries: List[Tuple[str, str, str]]
    use_task_identifier: bool = True

    @property
    def find_modality_type(self):  # Used in shell data preparation script
        all_entries = self.encoder_entries + self.decoder_entries
        ans = ""
        for entry in all_entries:
            ans = ans + ",".join(entry) + " "
        return ans


tasks = {}
tasks["tts"] = SpeechLMTask(
    encoder_entries=[("text", "g2p", "text"), ("utt2spk", "spk", "text")],
    decoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    target_entries=[("wav.scp", "codec", "kaldi_ark")],
)

tasks["bpe_tts"] = SpeechLMTask(
    encoder_entries=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    decoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    target_entries=[("wav.scp", "codec", "kaldi_ark")],
)

tasks["plain_tts"] = SpeechLMTask(
    encoder_entries=[("text", "g2p", "text")],
    decoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    target_entries=[("wav.scp", "codec", "kaldi_ark")],
)

tasks["plain_bpe_tts"] = SpeechLMTask(
    encoder_entries=[("text", "text_bpe", "text")],
    decoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    target_entries=[("wav.scp", "codec", "kaldi_ark")],
)

tasks["audiolm"] = SpeechLMTask(
    encoder_entries=[],
    decoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    target_entries=[("wav.scp", "codec", "kaldi_ark")],
)

tasks["textlm"] = SpeechLMTask(
    encoder_entries=[],
    decoder_entries=[("text", "text_bpe", "text")],
    target_entries=[("text", "text_bpe", "text")],
)

tasks["asr"] = SpeechLMTask(
    encoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    decoder_entries=[("text", "text_bpe", "text")],
    target_entries=[("text", "text_bpe", "text")],
)

############### END OF TASK DEFINITION ###############

# (3) Special token definition
# a. always reserve 256 slots for special tokens
#    0-31:    general special tokens
#    32-63:   modality identifier
#    64-127:  task identifier
#    128-255: reserved for future
# b. don't delete / modify it, otherwise the model trained
#    previously can become incompatible. New tokens can be
#    added - there are enough slots

special_tokens = [
    "<pad>",
    "<unk>",
    "<blank>",
    "<space>",
    "<continuous_placeholder>",
    "<sos/eos>",
    "<local_sos/eos>",
    "<unkown_task_identifer>",
]


def pad_until(token_list, until):
    assert until > len(token_list)
    for idx in range(len(token_list), until):
        token_list.append(f"<unused_token_{idx}>")
    return token_list


special_tokens = pad_until(special_tokens, 32)

for m in modalities.keys():
    special_tokens.append(f"<{m}_start/end>")
special_tokens = pad_until(special_tokens, 64)

for m in tasks.keys():
    special_tokens.append(f"<{m}_task>")
special_tokens = pad_until(special_tokens, 128)

special_tokens = pad_until(special_tokens, 256)

############### END OF SPECIAL TOKEN DEFINITION ###############
