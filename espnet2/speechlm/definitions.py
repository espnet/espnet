#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from dataclasses import dataclass
from typing import List, Tuple


# (1) Modality Definitions
@dataclass
class Modality:
    discrete: bool = True


MODALITIES = {}
# Discrete
MODALITIES["codec"] = Modality()
MODALITIES["ssl"] = Modality()
MODALITIES["codec_ssl"] = Modality()
MODALITIES["text_bpe"] = Modality()
MODALITIES["g2p"] = Modality()
MODALITIES["spk"] = Modality()
MODALITIES["class"] = Modality()
MODALITIES["bool"] = Modality()
MODALITIES["video_ssl"] = Modality()
MODALITIES["svs_lb"] = Modality()

MODALITIES["wav"] = Modality(discrete=False)
MODALITIES["text_emb"] = Modality(discrete=False)
MODALITIES["ssl_feat"] = Modality(discrete=False)

# END OF MODALITY DEFINITION #


# (2) Task Definition
@dataclass
class SpeechLMTaskTemplate:
    conditions: List[Tuple[str, str, str]]
    targets: List[Tuple[str, str, str]]
    use_task_identifier: bool = True

    @property
    def data_triplets(self):
        all_entries = self.conditions + self.targets
        return all_entries

    @property
    def data_triplets_string(self):
        ans = ""
        for entry in self.data_triplets:
            ans = ans + ",".join(entry) + " "
        return ans

    @property
    def condition_string(self):
        ans = ""
        for entry in self.conditions:
            ans = ans + ",".join(entry) + " "
        return ans

    @property
    def target_string(self):
        ans = ""
        for entry in self.targets:
            ans = ans + ",".join(entry) + " "
        return ans


SPEECHLM_TASKS = dict()

SPEECHLM_TASKS["textlm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "g2p", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["bpe_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["ssl_asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["mt"] = SpeechLMTaskTemplate(
    conditions=[("src_text", "text_bpe", "text")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["text2audio"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_emb", "kaldi_ark")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["visual_tts"] = SpeechLMTaskTemplate(
    conditions=[
        ("text", "g2p", "text"),
        ("utt2spk", "spk", "text"),
        ("video.scp", "video_ssl", "kaldi_ark"),
    ],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["vc"] = SpeechLMTaskTemplate(
    conditions=[("src_wav.scp", "codec", "kaldi_ark"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["ssl2codec"] = SpeechLMTaskTemplate(
    conditions=[("ssl_wav.scp", "ssl", "kaldi_ark"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["svs"] = SpeechLMTaskTemplate(
    conditions=[("label", "svs_lb", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["mt"] = SpeechLMTaskTemplate(
    conditions=[("src_text", "text_bpe", "text")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["st"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "ssl", "kaldi_ark")],
    targets=[("src_text", "text_bpe", "text"), ("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["se"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec", "kaldi_ark")],
    targets=[("spk1.scp", "codec", "kaldi_ark")],
)

# codec_ssl tasks:
SPEECHLM_TASKS["codec_ssl_asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["codec_ssl_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_plain_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_se"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("spk1.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_tse"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark"), ("utt2spk", "spk", "text")],
    targets=[("spk1.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["aac_codecssl"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["ag_codecssl"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
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

for m in MODALITIES.keys():
    special_tokens.append(f"<{m}_start/end>")
special_tokens = pad_until(special_tokens, 64)

for m in SPEECHLM_TASKS.keys():
    special_tokens.append(f"<{m}_task>")
special_tokens = pad_until(special_tokens, 128)

special_tokens = pad_until(special_tokens, 256)

# END OF SPECIAL TOKEN DEFINITION #
