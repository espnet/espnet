#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import torch
import torchaudio
from kaldiio import WriteHelper

from espnet2.speechlm.definitions import SPEECHLM_TASKS
from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer as LMTokenizer
from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.text.abs_tokenizer import AbsTokenizer as TextTokenizer
from espnet2.text.build_tokenizer import build_tokenizer


@dataclass
class AbsInferenceConfig:
    device: str
    nbest: int
    nq: int
    search_algo: str
    eos: int
    length_method: str
    maxlenratio: float = 10.0
    minlenratio: float = 0.0
    maxlen: int = 2048
    minlen: int = 0
    tokenizer: Union[LMTokenizer, TextTokenizer] = None
    mask: torch.Tensor = None


@dataclass
class TextInferenceConfig(AbsInferenceConfig):
    sampling_temperature: float = 1.0
    topk: int = 30


@dataclass
class SpeechInferenceConfig(AbsInferenceConfig):
    sampling_temperature: float = 1.0
    topk: int = 30


def build_inference_config(
    train_args,
    inference_config,
    device,
    nbest,
):
    config_map = {
        "text_bpe": TextInferenceConfig,
        "codec_ssl": SpeechInferenceConfig,
    }
    config_map = {m: obj for m, obj in config_map.items() if m in inference_config}

    retval = dict()
    for m, obj in config_map.items():
        config = inference_config[m]
        kwargs = {
            "device": device,
            "nbest": nbest,
            "nq": train_args.codec_token_in_use,
            "eos": train_args.token_list.index("<sos/eos>"),
            "search_algo": config["search_algo"],
            "length_method": config["length_method"],
        }

        # search algorithm
        if m == "text_bpe":
            assert config["search_algo"] in [
                "topk_sampling",
                "greedy_search",
                "teacher_force",
            ]
        elif m == "codec_ssl":
            assert config["search_algo"] in ["topk_sampling", "teacher_force"]

        if config["search_algo"] == "topk_sampling":
            kwargs["sampling_temperature"] = config["sampling_temperature"]
            kwargs["topk"] = config["topk"]

        # length method
        if config["length_method"] == "absolute":
            kwargs["maxlen"] = config["maxlen"]
            kwargs["minlen"] = config["minlen"]
        elif config["length_method"] == "relative":
            kwargs["maxlenratio"] = config["maxlenratio"]
            kwargs["minlenratio"] = config["minlenratio"]

        # tokenizer
        if m == "text_bpe":
            kwargs["tokenizer"] = build_tokenizer(
                train_args.subword_choice.replace("huggingface", "hugging_face"),
                train_args.subword_model,
            )
        elif m == "codec_ssl":
            kwargs["tokenizer"] = CodecTokenizer(**config["tokenizer"]).to(device)

        # mask
        if m == "text_bpe":
            kwargs["mask"] = build_mask(train_args, m).to(device)
        elif m == "codec_ssl":
            kwargs["mask"] = build_mask(train_args, m).to(device)

        retval[m] = obj(**kwargs)

    return retval


def build_mask(train_args, modality):

    if modality == "codec_ssl":
        codec_start, codec_end = train_args.token_bias["codec"]
        ssl_start, ssl_end = train_args.token_bias["ssl"]
    elif modality == "text_bpe":
        token_start, token_end = train_args.token_bias[modality]

    vocab_size = len(train_args.token_list)
    nq = train_args.codec_token_in_use
    identifier = f"<{modality}_start/end>"
    identifier = train_args.token_list.index(identifier)
    pad = train_args.token_list.index("<pad>")

    mask = torch.ones(nq, vocab_size).bool()
    mask[0, identifier] = False
    mask[1:, pad] = False  # always allow pad prediction in stream > 1

    if modality == "codec_ssl":
        mask[0, ssl_start:ssl_end] = False
        inc = (codec_end - codec_start) // (nq - 1)
        for i in range(1, nq):
            mask[i, codec_start + inc * (i - 1) : codec_start + inc * i] = False
    elif modality == "text_bpe":
        mask[0, token_start:token_end] = False

    return mask


class TaskOrientedWriter:
    """
    The writer to record the inference results when the SpeechLM work in *task* mode.
    In this mode, the task is well-defined, with a known number of segments and the
    corresponding modalities. The results are carefully checked with the task template
    to ensure compliance.
    """

    def __init__(
        self,
        train_args: Dict,
        task: str,
        output_dir: Path,
        rank: int = 0,
        inference_config: Dict[str, AbsInferenceConfig] = None,
    ):
        self.token_list = train_args.token_list
        self.token_bias = train_args.token_bias
        self.inference_config = inference_config
        self.task_template = SPEECHLM_TASKS[task]
        self.pad = self.token_list.index("<pad>")
        self.output_dir = output_dir
        self.task = task

        # Build writers
        output_dir.mkdir(parents=True, exist_ok=True)
        self.writers, self.token_writers = dict(), dict()
        for name, modality, _ in self.task_template.data_triplets:
            (output_dir / name).mkdir(parents=True, exist_ok=True)
            file_name = str(output_dir / name / f"rank{rank}_{name}")
            self.token_writers[name] = WriteHelper(
                f"ark,scp:{file_name}_token.ark,{file_name}_token.scp"
            )
            if modality in ["text_bpe", "codec_ssl", "spk"]:
                self.writers[name] = open(file_name, "w")

    @torch.no_grad()
    def write(self, uid, all_segments):
        all_segments[0] = all_segments[0][:, 2:]  # exclude <sos> and task identifier

        for m_idx, (name, modality, _) in enumerate(self.task_template.data_triplets):
            # (1) match modality
            segments = all_segments[m_idx]
            modality_ = segments[0][0, 0].item()
            modality_ = self.token_list[modality_]
            modality_ = modality_.removeprefix("<").removesuffix("_start/end>")
            if modality != modality_:
                raise ValueError(f"Expect {modality} but find {modality_}")

            # (2) identify detokenizer
            modality_ = "codec_ssl" if modality_ == "spk" else modality_
            config = self.inference_config[modality_]
            tokenizer = config.tokenizer

            is_prefill = isinstance(segments, torch.Tensor)
            for s_idx, segment in enumerate(segments):
                # (3.1) detokenize
                if is_prefill and s_idx > 0:
                    continue

                segment = segment[1:]
                segment = segment[segment[:, 0] != self.pad]

                if modality_ == "codec_ssl":
                    segment_codec = segment[:, 1:] - self.token_bias["codec"][0]
                    segment_codec = segment_codec.view(-1).contiguous()
                    detokenized = tokenizer.detokenize(
                        segment_codec.clone(),
                        n_codebook=config.nq - 1,
                    )

                    # Keep segment for saving, only substract ssl bias
                    segment = segment - self.token_bias["ssl"][0]

                elif modality_ == "text_bpe":
                    segment = segment[:, 0]
                    segment = segment.view(-1).contiguous()
                    detokenized = tokenizer.tokens2text(
                        [self.token_list[tok] for tok in segment]
                    ).strip()
                    segment = segment - self.token_bias["text_bpe"][0]

                # (3.2) write
                if is_prefill:
                    this_uid = uid.removeprefix(f"{self.task}_")
                else:
                    this_uid = f"{uid}_sample{s_idx}"

                self.token_writers[name][this_uid] = (
                    segment.flatten().int().cpu().numpy()
                )

                if modality_ == "codec_ssl":
                    audio_path = str(self.output_dir / name / f"{this_uid}.wav")
                    save_audio(audio_path, detokenized)
                    self.writers[name].write(f"{this_uid} {audio_path}\n")
                    logging.info(f"Save Audio for {this_uid}: {audio_path}")

                elif modality == "text_bpe":
                    self.writers[name].write(f"{this_uid} {detokenized}\n")
                    logging.info(f"Save Text for {this_uid}: {detokenized}")


class ChatOrientedWriter:
    """
    The writer to record the inference results when the SpeechLM work in *chat* mode.
    In this mode, there is no pre-defined tasks, the role, and modality of each segment
    is flexible. This is usually used in open-ended conversation, and we fully trust
    the modalities provided by the SpeechLM.
    """

    def __init__(
        self,
        train_args: Dict,
        task: str,
        output_dir: Path,
        rank: int = 0,
        inference_config: Dict[str, AbsInferenceConfig] = None,
    ):
        self.token_list = train_args.token_list
        self.token_bias = train_args.token_bias
        self.inference_config = inference_config
        self.output_dir = output_dir / "dialogue"
        self.task = task
        self.rank = rank

        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_name = str(self.output_dir / f"rank{rank}_dialogue")
        self.writer = WriteHelper(
            f"ark,scp:{file_name}_token.ark,{file_name}_token.scp"
        )
        self.buffer = []

    @torch.no_grad()
    def write(self, name, all_segments):
        dialogue = []

        for idx, segments in enumerate(all_segments):

            # name
            segment_name = f"{name}_segment{idx}"

            # segment
            if isinstance(segments, list):
                is_prefill = False
                assert len(segments) == 1
            else:
                is_prefill = True
            segment = segments[0]

            if idx == 0:  # exclude <sos> and <task_specifier>
                segment = segment[2:]

            # role
            role = segment[0][0]
            if role == 8:
                role = "system"
            elif role == 9:
                role = "user"
            elif role == 10:
                role = "assistant"
            else:
                raise ValueError("Invalid role token")

            modality = segment[1][0]
            modality = self.token_list[modality]
            modality = modality.removeprefix("<").removesuffix("_start/end>")
            modality = "codec_ssl" if modality == "spk" else modality

            segment = segment[2:]  # exclude role and modality token
            segment = segment[segment[:, 0] != 0]  # exclude padding

            # detokenize
            if modality == "codec_ssl":
                segment_codec = segment[:, 1:] - self.token_bias["codec"][0]
                segment_codec = segment_codec.view(-1).contiguous()
                tokenizer = self.inference_config[modality].tokenizer
                detokenized = tokenizer.detokenize(
                    segment_codec.clone(),
                    n_codebook=self.inference_config[modality].nq - 1,
                )

                segment = segment - self.token_bias["ssl"][0]

            elif modality == "text_bpe":
                segment = segment[:, 0]
                segment = segment.view(-1).contiguous()
                tokenizer = self.inference_config[modality].tokenizer
                detokenized = tokenizer.tokens2text(
                    [self.token_list[tok] for tok in segment]
                ).strip()
                segment = segment - self.token_bias["text_bpe"][0]

            else:
                raise NotImplementedError(
                    f"modality detokenization on {modality} is not supported yet."
                )

            # write
            self.writer[segment_name] = segment.int().flatten().cpu().numpy()

            if modality == "codec_ssl":
                audio_path = str(self.output_dir / f"{segment_name}.wav")
                save_audio(audio_path, detokenized)
                detokenized = audio_path

            dialogue.append([role, modality, is_prefill, detokenized])
            logging.info(
                f"Index: {idx}, Role={role}, Modality={modality}, "
                f"is_prefill={is_prefill}, Content={detokenized}"
            )

        self.buffer.append({name: dialogue})
        if len(self.buffer) % 1 == 0:  # save results periodically
            json_writer = open(self.output_dir / f"rank{self.rank}_dialogue.json", "wb")
            json_writer.write(
                json.dumps(
                    self.buffer, indent=4, ensure_ascii=False, sort_keys=False
                ).encode("utf_8")
            )


def parse_sequence(dec_seq, token_list, mode="task", inference_last_segment=False):
    """
    Parse the sequence into multiple segments for inference.
    Args:
        dec_seq (torch.Tensor): The sequence to be parsed, of size [B, T, nq].
        mode (str): The mode of parsing, either "task" or "chat".

    returns:
        all_segments (list): A list of segments, each two items:
            - segments, List of torch.Tensor
            - is prefill, bool
    """

    # (1) find all segments start from a modality specifier
    indices = dec_seq[0, :, 0]
    indices = (indices >= 32) & (indices < 64)
    indices = indices.nonzero(as_tuple=True)[0]

    # (1.1) if the chat model, add the start with role token
    if mode == "chat":
        indices -= 1

    # (1.2) the first segment start from 0; add the last index
    indices[0] = 0
    indices = torch.nn.functional.pad(indices, (0, 1), value=dec_seq.size(1))

    # (2) find the task
    task_token = dec_seq[0, 1, 0].item()
    task_token = token_list[task_token].removeprefix("<").removesuffix("_task>")
    task_template = SPEECHLM_TASKS[task_token]

    if mode == "task":
        assert len(task_template.data_triplets) == len(indices) - 1

    # (2) each segment is either a prefill or a target
    all_segments = []
    is_prefills = []
    for idx, (start, end) in enumerate(zip(indices[:-1], indices[1:])):
        # (2.1) get the segment
        segment = dec_seq[:, start:end, :]
        all_segments.append(segment)

        # (2.2) check if the segment is a prefill or a target
        if mode == "task":
            data_triplet = task_template.data_triplets[idx]
            if data_triplet in task_template.targets:
                is_prefills.append(False)
            else:
                is_prefills.append(True)
        else:
            if segment[0, 0, 0] == 10:  # role token, <assistant_output>
                is_prefills.append(False)
            else:
                is_prefills.append(True)

    if inference_last_segment:
        for n in range(len(is_prefills) - 1):
            is_prefills[n] = True

    return all_segments, is_prefills


def save_audio(path, audio):
    torchaudio.save(
        path,
        audio.view(1, -1).cpu(),
        sample_rate=16000,
        bits_per_sample=16,
        encoding="PCM_S",
    )
