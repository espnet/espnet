#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from dataclasses import dataclass
from typing import List, Tuple

# Users are encouraged to read our document to understand the definitions in this file:
# https://github.com/jctian98/espnet/blob/master/egs2/TEMPLATE/speechlm1/README.md


# (1) Modality definition
# a. Modality could be either discrete or continuous, depending on how the LM
#    processes the data in that modality.
# b. Data from discrete modalities (e.g., text_bpe, codec) will experience an
#    embedding process before being fed into LM.
# c. Data from continuous modalities (e.g., text_emb) will be projected by an
#    MLP before being processed by the LM.
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
MODALITIES["image"] = Modality()

# continuous
MODALITIES["wav"] = Modality(discrete=False)
MODALITIES["text_emb"] = Modality(discrete=False)
MODALITIES["ssl_feat"] = Modality(discrete=False)

# dialogue
MODALITIES["dialogue"] = Modality()

# END OF MODALITY DEFINITION #


# (2) Task Definition
# a. The task definition contains conditions and targets, both are represented
#    by a list of "triplets", as described in README file.
# b. The conditions specifies the input of that task. e.g., audio of ASR
# c. The targets specifies the output of that task. e.g., text of ASR.
@dataclass
class SpeechLMTaskTemplate:
    conditions: List[Tuple[str, str, str]]
    targets: List[Tuple[str, str, str]]

    @property
    def data_triplets(self):
        all_entries = self.conditions + self.targets
        return all_entries

    @property
    def n_conditions(self):
        return len(self.conditions)

    @property
    def n_targets(self):
        return len(self.targets)

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


# Here we only keep limited number of task definitions. For more examples, users
# can either define by their own, or refer to our experimental file below. Note
# some of those task template are not well justified.
# https://github.com/jctian98/espnet/blob/speechlm3/espnet2/speechlm/definitions.py
SPEECHLM_TASKS = dict()

# Task-ID: 64
# Auto-regressive prediction over text_bpe tokens, aka, standard text LM task
SPEECHLM_TASKS["textlm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("text", "text_bpe", "text")],
)

# Task-ID: 65
# Auto-regressive prediction over audio sequence, where audio is represented by
# codec_ssl modality.
SPEECHLM_TASKS["codec_ssl_audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

# Task-ID: 66
# Speech Recognition task, predict text based on audio input.
# Note audio is represented by codec_ssl modality
SPEECHLM_TASKS["codec_ssl_asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

# Task-ID: 67
# Text-to-Speech task, predict audio based on text and speaker prompt.
# Note audio is represented by codec_ssl modality.
SPEECHLM_TASKS["codec_ssl_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

# Task-ID: 68
# Free-form conversational data with textual content only.
# The exact content is described in each dialogue json files.
SPEECHLM_TASKS["text_dialogue"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("dialogue", "dialogue", "dialogue_json")],
)

# Task-ID: 69
# Free-form conversational data with both audio and text content.
# The exact content is described in each dialogue json files.
SPEECHLM_TASKS["audio_dialogue"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("dialogue", "dialogue", "dialogue_json")],
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
# c. detailed explanation for frequently special tokens:
#    <pad>: padding tokens. These tokens is for padding only and will not participate
#           loss computing.
#    <sos/eos>: start-of-sentence/end-of-senetence. Each sequence always starts and
#           ends with this token.
#    <system_prompt>, <user_input>, <assistant_output>: role tokens in chat template.
#    <eou>: end-of-utternace, end of an utterance (or a short segment) of certain
#           modality. Usually used as a termination signal in decoding.
#    modality tokens, e.g., <text_bpe_start/end>: the token to indicate modality. This
#           token is always in the first place of a segment. e.g.,
#           <text_bpe_start/end>, text_token1, ..., text_tokenN
#    task tokens, e.g., <asr_task>: the indicator of a certain task. This token is
#           always in the second place of a whole sequence, following <sos/eos>.
#    Other special tokens are deprecated or not in frequent usage.
#    See use case in:
#    https://github.com/jctian98/espnet/tree/speechlm3/egs2/
#    TEMPLATE/speechlm1#example-sequence
special_tokens = [
    "<pad>",
    "<unk>",
    "<blank>",
    "<space>",
    "<continuous_placeholder>",
    "<sos/eos>",
    "<local_sos/eos>",
    "<unkown_task_identifer>",
    "<system_prompt>",
    "<user_input>",
    "<assistant_output>",
    "<eou>",
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
