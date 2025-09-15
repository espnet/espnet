"""
Trainer module for language identification and language embedding extraction.
"""

import logging
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.torch_utils.device_funcs import to_device
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions


class LIDTrainer(Trainer):
    """Trainer designed for LID, adapted from spk_trainer.py"""

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @torch.no_grad()
    @typechecked
    def extract_embed_lid(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        output_dir: str,
        custom_bs: int,
        idx2lang: Dict[int, str],
        extract_embd: bool = False,
        checkpoint_interval: int = 1000,
        resume: bool = True,
        lang_to_embds_dic: Dict[str, List[np.ndarray]] = None,
        save_embd_per_utt: bool = False,
        max_num_utt_per_lang: Optional[int] = None,
        lang_counter_dic: Optional[Dict[str, int]] = None,
    ) -> None:
        r"""Extract LIDs and language embeddings for each utterance in the dataset.

        By default, this method performs language identification (LID) for
        each utterance. If `extract_embd=True`, it also extracts normalized language
        embeddings.

        lang_embd_dic: {utt_id: lang_embd}, the language embedding for a specific
        utterance, this is used for temporary saving the language embedding of each
        utterance, and will be written to the disk every `checkpoint_interval`
        utterances.
        lang_to_embds_dic: {lang: [utt1 embd, utt1 embd ...]}, the language embedding
        for the utterances corresponding to each language, if set `extract_embd` to
        True, this will be defaultly used, this will not be written to the dist, but
        will be (in bin/lid_inference.py) used for calculating the average language
        embedding for each language, and plotting the tsne plot.

        Saved results:
        - lang_id_dic: {utt_id: predicted_lang}, mapping from utterance ID to
                                predicted language ID.
        - lang_embd_dic (optional): {utt_id: lang_embd}, temporary in-memory
                                    storage of per-utterance language embeddings.
                                    Saved to disk every `checkpoint_interval`
                                    utterances if `save_embd_per_utt=True`.
        - lang_to_embds_dic (optional): {lang: [embd_utt1, embd_utt2, ...]},
                                        mapping from language ID to a list of
                                        embeddings from all utterances predicted
                                        or labeled with that language. This is
                                        not written to disk by this function, but
                                        is used downstream (e.g., in
                                        `bin/lid_inference.py`) for computing
                                        language-level average embeddings or
                                        generating t-SNE visualizations.

        Notes:
        - All extracted embeddings are L2-normalized.
        - The function supports distributed inference using torch.distributed.
        - Supports resume functionality by skipping already processed utterances
          based on existing output files.
        - Limits the number of utterances per language if `max_num_utt_per_lang`
          is specified.
        """
        # Extract language embedding and lids.
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        model.eval()
        if extract_embd:
            # {utt_id: lang_embd}, the language embedding for a specific utterance
            lang_embd_dic = {}
        lang_id_dic = {}  # {utt_id: lang_id}

        utt_id_list = []
        utt_id_whole_list = []
        speech_list = []
        speech_length_list = []
        lid_label_list = []

        if distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        idx = 0
        step = 0  # for save middle results
        num_recheck = 0
        if resume:
            skip_utts = set()
            if os.path.exists(f"{output_dir}/lids{rank}"):
                with open(f"{output_dir}/lids{rank}", "r") as f:
                    for line in f:
                        utt_id, lid = line.strip().split()
                        skip_utts.add(utt_id)
            logging.info(
                f"[Rank {rank}] Resume: {len(skip_utts)} utterances found in "
                f"{output_dir}/lids{rank}"
            )

        for utt_id, batch in iterator:
            if max_num_utt_per_lang is not None and lang_counter_dic is not None:
                num_langs_reach_max_num = 0
                for count in lang_counter_dic.values():
                    if count >= max_num_utt_per_lang:
                        num_langs_reach_max_num += 1
                if num_langs_reach_max_num == len(lang_counter_dic):
                    logging.info(
                        f"[Rank {rank}] All languages reach max_num_utt_per_lang: "
                        f"{max_num_utt_per_lang}."
                    )
                    break

            assert isinstance(batch, dict), type(batch)
            for _utt_id, _speech, _speech_length, _lid_label in zip(
                utt_id,
                batch["speech"],
                batch["speech_lengths"],
                batch.get(
                    "lid_labels", [None] * len(utt_id)
                ),  # lid label is the index, not the iso3 code.
            ):
                if resume:
                    if _utt_id in skip_utts:
                        if num_recheck % checkpoint_interval == 0 and num_recheck > 0:
                            step += 1
                            logging.info(
                                f"[Rank {rank}] Skip utterance "
                                f"{num_recheck - checkpoint_interval}-{num_recheck}."
                            )
                        num_recheck += 1
                        continue
                if _utt_id not in utt_id_whole_list:
                    # Restrict the number of utterances per language when plotting tsne
                    if (
                        max_num_utt_per_lang is not None
                        and lang_counter_dic is not None
                        and _lid_label is not None
                    ):
                        if (
                            lang_counter_dic[idx2lang[_lid_label.item()]]
                            >= max_num_utt_per_lang
                        ):
                            logging.info(
                                f"[Rank {rank}] Language {idx2lang[_lid_label.item()]} "
                                f"reach max_num_utt_per_lang: {max_num_utt_per_lang}."
                            )
                            continue
                        else:
                            lang_counter_dic[idx2lang[_lid_label.item()]] += 1

                    utt_id_whole_list.append(_utt_id)
                    if idx % world_size == rank:
                        utt_id_list.append(_utt_id)
                        speech_list.append(_speech)
                        speech_length_list.append(_speech_length)
                        lid_label_list.append(_lid_label)
                    idx += 1

                    if len(utt_id_list) == custom_bs:
                        try:
                            speech_list = torch.stack(
                                speech_list, dim=0
                            )  # (bs, t), t is the length of the speech
                            speech_length_list = torch.stack(
                                speech_length_list, dim=0
                            )  # (bs,)
                        except (
                            RuntimeError
                        ):  # for last few batches, pad to the same length
                            max_len = max(s.size(0) for s in speech_list)
                            speech_list = torch.stack(
                                [
                                    F.pad(s, (0, max_len - s.size(0)))
                                    for s in speech_list
                                ],
                                dim=0,
                            )
                            speech_length_list = torch.stack(speech_length_list, dim=0)

                        speech_list = to_device(
                            speech_list, "cuda" if ngpu > 0 else "cpu"
                        )
                        speech_length_list = to_device(
                            speech_length_list, "cuda" if ngpu > 0 else "cpu"
                        )

                        lang_embds, pred_lids = model(
                            speech=speech_list,
                            speech_lengths=speech_length_list,
                            lid_labels=None,
                            extract_embd=True,
                        )  # [batch_size, dim], [batch_size]

                        if extract_embd:
                            lang_embds = F.normalize(lang_embds, p=2, dim=1)
                        pred_lids = [idx2lang[lid.item()] for lid in pred_lids]

                        for uid, _lang_embd, _pred_lid, _lid_label_target in zip(
                            utt_id_list, lang_embds, pred_lids, lid_label_list
                        ):
                            if extract_embd and _lid_label_target is not None:
                                # Save lang to embeddings dictionary
                                _lang_embd_numpy = _lang_embd.detach().cpu().numpy()
                                target_lid = idx2lang[_lid_label_target.item()]
                                lang_to_embds_dic[target_lid].append(_lang_embd_numpy)
                                lang_embd_dic[uid] = _lang_embd_numpy

                            lang_id_dic[uid] = _pred_lid

                        # save every `checkpoint_interval` utterances
                        if len(lang_id_dic) >= checkpoint_interval:
                            if extract_embd and save_embd_per_utt:
                                # save each middle step results to different files
                                np.savez(
                                    output_dir
                                    + f"/embeddings{rank + world_size * step}",
                                    **lang_embd_dic,
                                )

                            with open(f"{output_dir}/lids{rank}", "a") as f:
                                # save all middle step results to the same file
                                for uid, lid in lang_id_dic.items():
                                    f.write(f"{uid} {lid}\n")
                            logging.info(
                                f"[Rank {rank}] Saved {len(lang_id_dic)} utts at "
                                f"step {step}"
                            )

                            if (
                                max_num_utt_per_lang is not None
                                and lang_counter_dic is not None
                            ):
                                logging.info(
                                    f"[Rank {rank}] Current lang_counter_dic: "
                                    f"{lang_counter_dic}"
                                )

                            if extract_embd:
                                lang_embd_dic.clear()
                            lang_id_dic.clear()
                            step += 1

                        utt_id_list = []
                        speech_list = []
                        speech_length_list = []
                        lid_label_list = []

        if len(utt_id_list) != 0:
            # Process the remaining utterances in the last batch
            try:
                speech_list = torch.stack(speech_list, dim=0)
                speech_length_list = torch.stack(speech_length_list, dim=0)  # (bs,)
            except RuntimeError:  # for last few batches, pad to the same length
                max_len = max(s.size(0) for s in speech_list)
                speech_list = torch.stack(
                    [F.pad(s, (0, max_len - s.size(0))) for s in speech_list], dim=0
                )
                speech_length_list = torch.stack(speech_length_list, dim=0)

            speech_list = to_device(speech_list, "cuda" if ngpu > 0 else "cpu")
            speech_length_list = to_device(
                speech_length_list, "cuda" if ngpu > 0 else "cpu"
            )

            lang_embds, pred_lids = model(
                speech=speech_list,
                speech_lengths=speech_length_list,
                lid_labels=None,
                extract_embd=True,
            )  # [batch_size, dim], [batch_size]

            if extract_embd:
                lang_embds = F.normalize(lang_embds, p=2, dim=1)
            pred_lids = [idx2lang[lid.item()] for lid in pred_lids]

            for uid, _lang_embd, _pred_lid, _lid_label_target in zip(
                utt_id_list, lang_embds, pred_lids, lid_label_list
            ):
                if extract_embd and _lid_label_target is not None:
                    # Save lang to embeddings dictionary
                    _lang_embd_numpy = _lang_embd.detach().cpu().numpy()
                    target_lid = idx2lang[_lid_label_target.item()]
                    lang_to_embds_dic[target_lid].append(_lang_embd_numpy)
                    lang_embd_dic[uid] = _lang_embd_numpy

                lang_id_dic[uid] = _pred_lid

        if len(lang_id_dic) != 0:
            # Save the last results
            if extract_embd and save_embd_per_utt:
                np.savez(
                    output_dir + f"/embeddings{rank + world_size * step}",
                    **lang_embd_dic,
                )

            with open(f"{output_dir}/lids{rank}", "a") as f:
                # save all middle step results to the same file
                for uid, lid in lang_id_dic.items():
                    f.write(f"{uid} {lid}\n")
