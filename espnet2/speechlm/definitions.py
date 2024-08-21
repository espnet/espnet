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
    data_type: str = ("kaldi_ark",)


modalities = {}
# Discrete
modalities["codec"] = Modality()
modalities["ssl"] = Modality()
modalities["text_bpe"] = Modality(data_type="text")
modalities["g2p"] = Modality(data_type="text")
modalities["spk"] = Modality(data_type="text")
# Continuous
modalities["wav"] = Modality(discrete=False)
modalities["text_emb"] = Modality(discrete=False)
modalities["ssl_feat"] = Modality(discrete=False)

# END OF MODALITY DEFINITION #


# (2) Task Definition
# a. usually we will place a task identifier in the begining.
#    however, when we want to specify the task by natual langauge,
#    we don't use that identifier.
# b. encoder_entries: the entires that should feed to encoder, which
#    is a list of tuple (file_name, entry_modality, data_type).
# c. decoder_entries: similar to encoder_entries, but is fed to decoder.
# d. in decoder-only format, encoder_entries and decoder_entries are merged
# e. target_entries: entries that the loss computed on. Usually same as
#    the decoder_entries.
# f. file_name, the expected file name in original data folder. e.g., wav.scp
#    entry_modality: the modality defined above, which will be used to determine
#      how this data will be pre-processed before training. e.g., codec tokenization
#    data_type: it determines how the data will be loaded during training.
#    E.g., in TTS, the wave files are indexed with wav.scp, it will experence codec
#      tokenization and then loaded as kaldi_ark -> (wav.scp, codec, kaldi_ark)
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

tasks["plain_tts"] = SpeechLMTask(
    encoder_entries=[("text", "g2p", "text")],
    decoder_entries=[("wav.scp", "codec", "kaldi_ark")],
    target_entries=[("wav.scp", "codec", "kaldi_ark")],
)

# END OF TASK DEFINITION #

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

# END OF SPECIAL TOKEN DEFINITION #
