"""
Trainer module for speaker recognition.
In speaker recognition (embedding extractor training/inference),
calculating validation loss in closed set is not informative since
generalization in unseen utterances from known speakers are good in most cases.
Thus, we measure open set equal error rate (EER) using unknown speakers by
overriding validate_one_epoch.
"""

import re
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from a_dcf import a_dcf
from tqdm import tqdm
from typeguard import typechecked

from espnet2.torch_utils.device_funcs import to_device
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.eer import (
    ComputeErrorRates,
    ComputeMinDcf,
    SASVCostModel,
    tuneThresholdfromScore,
)

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


class SpkTrainer(Trainer):
    """Trainer designed for speaker recognition.

    Training will be done as closed set classification.
    Validation will be open set EER calculation.
    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @torch.no_grad()
    @typechecked
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        model.eval()

        scores = []
        labels = []
        spk_embd_dic = {}
        bs = 0

        embed_avg = (
            False  # use speech, speech2, and speech3 as enrollment and speech4 as test
        )
        sasv = False  # spoofing aware speaker verification task

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        for utt_id, batch in iterator:
            pattern = r"[DE]_\d{4}\*[DE]_\d{10}"
            if any(re.match(pattern, uid) for uid in utt_id):
                sasv = True

            # Set iterator_stop to 1 to indicate the loop should break after this iteration
            iterator_stop.fill_(1)

            if distributed:
                # Synchronize iterator_stop across all processes
                torch.distributed.all_reduce(
                    iterator_stop, op=torch.distributed.ReduceOp.SUM
                )
                if iterator_stop > 0:
                    break
            else:
                # In non-distributed mode, break after setting iterator_stop
                break

        # Synchronize sasv across all processes
        if distributed:
            sasv_tensor = torch.tensor(
                [1 if sasv else 0], device="cuda" if ngpu > 0 else "cpu"
            )
            torch.distributed.all_reduce(sasv_tensor, op=torch.distributed.ReduceOp.MAX)
            sasv = bool(sasv_tensor.item())

        # Reset iterator_stop for subsequent use
        if distributed:
            iterator_stop.zero_()
            # Ensure iterator_stop is synchronized after resetting
            torch.distributed.all_reduce(
                iterator_stop, op=torch.distributed.ReduceOp.SUM
            )

        # trials list
        all_trials_list = []

        task_token = None

        for utt_id, batch in tqdm(iterator):
            utt_id_list = []
            speech_list = []
            bs = max(bs, len(utt_id))
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]

            # check if batch has speech3 and speech4
            if "speech3" in batch:
                assert "speech4" in batch
                embed_avg = True

            assert isinstance(batch, dict), type(batch)

            if embed_avg:
                for _utt_id, _speech, _speech2, _speech3, _speech4 in zip(
                    utt_id,
                    batch["speech"],
                    batch["speech2"],
                    batch["speech3"],
                    batch["speech4"],
                ):
                    _utt_id_1, _utt_id_2 = _utt_id.split("*")
                    # speech, speech2 and speech3 are enrollment utterances for the speaker whose
                    # speakerid is _utt_id_1 and speech4 is the test utterance
                    # for the purpose of enrollment, the uttid for each enrollment utterance will be
                    # recorded as _utt_id_1-first, _utt_id_1-second, and _utt_id_1-third
                    if (_utt_id_1 + "-first") not in utt_id_list:
                        utt_id_list.append(_utt_id_1 + "-first")
                        speech_list.append(
                            to_device(_speech, "cuda" if ngpu > 0 else "cpu")
                        )
                    if (_utt_id_1 + "-second") not in utt_id_list:
                        utt_id_list.append(_utt_id_1 + "-second")
                        speech_list.append(
                            to_device(_speech2, "cuda" if ngpu > 0 else "cpu")
                        )
                    if (_utt_id_1 + "-third") not in utt_id_list:
                        utt_id_list.append(_utt_id_1 + "-third")
                        speech_list.append(
                            to_device(_speech3, "cuda" if ngpu > 0 else "cpu")
                        )
                    if _utt_id_2 not in utt_id_list:
                        utt_id_list.append(_utt_id_2)
                        speech_list.append(
                            to_device(_speech4, "cuda" if ngpu > 0 else "cpu")
                        )
            else:
                for _utt_id, _speech, _speech2 in zip(
                    utt_id, batch["speech"], batch["speech2"]
                ):
                    _utt_id_1, _utt_id_2 = _utt_id.split("*")
                    if _utt_id_1 not in utt_id_list:
                        utt_id_list.append(_utt_id_1)
                        speech_list.append(
                            to_device(_speech, "cuda" if ngpu > 0 else "cpu")
                        )
                    if _utt_id_2 not in utt_id_list:
                        utt_id_list.append(_utt_id_2)
                        speech_list.append(
                            to_device(_speech2, "cuda" if ngpu > 0 else "cpu")
                        )

            assert len(utt_id_list) == len(speech_list)

            # extract speaker embeddings.
            n_utt = len(utt_id_list)
            for ii in range(0, n_utt, bs):
                _utt_ids = utt_id_list[ii : ii + bs]
                _speechs = speech_list[ii : ii + bs]
                _speechs = torch.stack(_speechs, dim=0)
                org_shape = (_speechs.size(0), _speechs.size(1))
                _speechs = _speechs.flatten(0, 1)
                _speechs = to_device(_speechs, "cuda" if ngpu > 0 else "cpu")

                if task_token is None:
                    task_tokens = None
                else:
                    task_tokens = to_device(
                        task_token.repeat(_speechs.size(0)),
                        "cuda" if ngpu > 0 else "cpu",
                    ).unsqueeze(1)
                spk_embds = model(
                    speech=_speechs,
                    spk_labels=None,
                    extract_embd=True,
                    task_tokens=task_tokens,
                )

                spk_embds = F.normalize(spk_embds, p=2, dim=1)
                spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                for _utt_id, _spk_embd in zip(_utt_ids, spk_embds):
                    if embed_avg:
                        if (
                            _utt_id.endswith("-first")
                            or _utt_id.endswith("-second")
                            or _utt_id.endswith("-third")
                        ):
                            spkID = _utt_id.split("-")[0]
                            if spkID in spk_embd_dic:
                                spk_embd_dic[spkID] = torch.cat(
                                    (spk_embd_dic[spkID], _spk_embd), dim=0
                                )
                            else:  # first time we see this speaker
                                spk_embd_dic[spkID] = _spk_embd
                        else:  # test utterance
                            spk_embd_dic[_utt_id] = _spk_embd
                    else:  # not embed_avg
                        spk_embd_dic[_utt_id] = _spk_embd

            del utt_id_list
            del speech_list

        # Compute the average embedding for each speaker
        if embed_avg:
            for spkID in spk_embd_dic:
                spk_embd_dic[spkID] = torch.mean(
                    spk_embd_dic[spkID], dim=0, keepdim=True
                )

        # calculate similarity scores
        for utt_id, batch in iterator:
            batch["spk_labels"] = to_device(
                batch["spk_labels"], "cuda" if ngpu > 0 else "cpu"
            )

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            all_trials_list.extend(utt_id)

            for _utt_id in utt_id:
                _utt_id_1, _utt_id_2 = _utt_id.split("*")
                if not sasv:
                    score = torch.cdist(
                        spk_embd_dic[_utt_id_1], spk_embd_dic[_utt_id_2]
                    )
                    score = -1.0 * torch.mean(score)
                else:
                    score = F.cosine_similarity(
                        spk_embd_dic[_utt_id_1], spk_embd_dic[_utt_id_2], dim=1
                    )
                scores.append(score.view(1))  # 0-dim to 1-dim tensor for cat
            labels.append(batch["spk_labels"])

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        torch.cuda.empty_cache()

        scores = torch.cat(scores).type(torch.float32)
        labels = torch.cat(labels).type(torch.int32).flatten()

        if distributed:
            # get the number of trials assigned on each GPU
            length = to_device(
                torch.tensor([labels.size(0)], dtype=torch.int32), "cuda"
            )
            lengths_all = [
                to_device(torch.zeros(1, dtype=torch.int32), "cuda")
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(lengths_all, length)

            scores_all = [
                to_device(torch.zeros(i, dtype=torch.float32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(scores_all, scores)
            scores = torch.cat(scores_all)

            labels_all = [
                to_device(torch.zeros(i, dtype=torch.int32), "cuda")
                for i in lengths_all
            ]
            torch.distributed.all_gather(labels_all, labels)
            labels = torch.cat(labels_all)
            # rank = torch.distributed.get_rank()
            torch.distributed.barrier()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # gather all trials list
        if distributed:
            gathered_trials = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_trials, all_trials_list)
            trials_list = [item for sublist in gathered_trials for item in sublist]
        else:
            trials_list = all_trials_list

        if sasv == False:
            # calculate statistics in target and nontarget classes.
            n_trials = len(scores)
            scores_trg = []
            scores_nontrg = []
            for _s, _l in zip(scores, labels):
                if _l == 1:
                    scores_trg.append(_s)
                elif _l == 0:
                    scores_nontrg.append(_s)
                else:
                    raise ValueError(f"{_l}, {type(_l)}")
            trg_mean = float(np.mean(scores_trg))
            trg_std = float(np.std(scores_trg))
            nontrg_mean = float(np.std(scores_nontrg))
            nontrg_std = float(np.std(scores_nontrg))

            # exception for collect_stats.
            if len(scores) == 1:
                reporter.register(stats=dict(eer=1.0, mindcf=1.0))
                return

            # predictions, ground truth, and the false acceptance rates to calculate
            results = tuneThresholdfromScore(scores, labels, [1, 0.1])
            eer = results[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

            # p_target, c_miss, and c_falsealarm in NIST minDCF calculation
            p_trg, c_miss, c_fa = 0.05, 1, 1
            mindcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_trg, c_miss, c_fa)

            reporter.register(
                stats=dict(
                    eer=eer,
                    mindcf=mindcf,
                    n_trials=n_trials,
                    trg_mean=trg_mean,
                    trg_std=trg_std,
                    nontrg_mean=nontrg_mean,
                    nontrg_std=nontrg_std,
                )
            )

        else:
            idx2class = {}
            idx2class[0] = "target"
            idx2class[1] = "nontarget"
            idx2class[2] = "spoof"
            # calculate statistics in target, nontarget, and spoof classes.
            n_trials = len(scores)
            scores_trg = []
            scores_nontrg = []
            scores_spoof = []
            for _s, _l in zip(scores, labels):
                if _l == 0:
                    scores_trg.append(_s)
                elif _l == 1:
                    scores_nontrg.append(_s)
                elif _l == 2:
                    scores_spoof.append(_s)
                else:
                    raise ValueError(f"{_l}, {type(_l)}")
            trg_mean = float(np.mean(scores_trg))
            trg_std = float(np.std(scores_trg))
            nontrg_mean = float(np.std(scores_nontrg))
            nontrg_std = float(np.std(scores_nontrg))
            spoof_mean = float(np.mean(scores_spoof))
            spoof_std = float(np.std(scores_spoof))

            # exception for collect_stats.
            if len(scores) == 1:
                reporter.register(stats=dict(eer=1.0, mindcf=1.0))
                return

            # write scores to file for a-dcf calculation
            # format should be:
            # # <speaker_id> <utterance_id> <score> <trial type>
            # where speaker_id and utterance_id are obtained from the utt_id in format <speaker_id>*<utterance_id>
            # and trial type is 0 for target, 1 for nontarget, and 2 for spoof but should be mapped to string using idx2class
            print(f"num trials: {len(trials_list)}")
            print(f"num scores: {len(scores)}")
            assert len(trials_list) == len(scores)
            with open("scores.txt", "w") as f:
                for i in range(len(trials_list)):
                    f.write(
                        f"{trials_list[i].split('*')[0]} {trials_list[i].split('*')[1]} {scores[i]} {idx2class[labels[i]]}\n"
                    )

            # calculate a-dcf
            adcf_results = a_dcf.calculate_a_dcf(
                "scores.txt", cost_model=SASVCostModel()
            )
            adcf = adcf_results["min_a_dcf"]

            reporter.register(
                stats=dict(
                    a_dcf=adcf,
                    n_trials=n_trials,
                    trg_mean=trg_mean,
                    trg_std=trg_std,
                    nontrg_mean=nontrg_mean,
                    nontrg_std=nontrg_std,
                    spoof_mean=spoof_mean,
                    spoof_std=spoof_std,
                )
            )

        # added to reduce GRAM usage. May have minor speed boost when
        # this line is commented in case GRAM is not fully used.
        torch.cuda.empty_cache()

    @classmethod
    @torch.no_grad()
    @typechecked
    def extract_embed(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        output_dir: str,
        custom_bs: int,
        average: bool = False,
        embed_avg: bool = False,  # TODO: cleanup to avoid duplication
    ) -> None:
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        model.eval()
        spk_embd_dic = {}

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        # iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        # fill dictionary with speech samples
        utt_id_list = []
        utt_id_whole_list = []
        speech_list = []
        task_token = None

        if distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        idx = 0

        if not embed_avg:
            for utt_id, batch in iterator:
                if "task_tokens" in batch:
                    task_token = batch["task_tokens"][0]

                assert isinstance(batch, dict), type(batch)
                for _utt_id, _speech, _speech2 in zip(
                    utt_id, batch["speech"], batch["speech2"]
                ):
                    _utt_id_1, _utt_id_2 = _utt_id.split("*")
                    if _utt_id_1 not in utt_id_whole_list:
                        utt_id_whole_list.append(_utt_id_1)
                        if idx % world_size == rank:
                            utt_id_list.append(_utt_id_1)
                            speech_list.append(_speech)

                        if len(utt_id_list) == custom_bs:
                            speech_list = torch.stack(speech_list, dim=0)
                            org_shape = (speech_list.size(0), speech_list.size(1))
                            speech_list = speech_list.flatten(0, 1)
                            speech_list = to_device(
                                speech_list, "cuda" if ngpu > 0 else "cpu"
                            )
                            if task_token is None:
                                task_tokens = None
                            else:
                                task_tokens = to_device(
                                    task_token.repeat(speech_list.size(0)),
                                    "cuda" if ngpu > 0 else "cpu",
                                ).unsqueeze(1)
                            spk_embds = model(
                                speech=speech_list,
                                spk_labels=None,
                                extract_embd=True,
                                task_tokens=task_tokens,
                            )
                            # removed to be use magnitude in qmf
                            # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                                if average:
                                    spk_embd_dic[uid] = (
                                        _spk_embd.mean(0).detach().cpu().numpy()
                                    )
                                else:
                                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()

                            utt_id_list = []
                            speech_list = []

                        idx += 1
                    if _utt_id_2 not in utt_id_whole_list:
                        utt_id_whole_list.append(_utt_id_2)
                        if idx % world_size == rank:
                            utt_id_list.append(_utt_id_2)
                            speech_list.append(_speech2)

                        if len(utt_id_list) == custom_bs:
                            speech_list = torch.stack(speech_list, dim=0)
                            org_shape = (speech_list.size(0), speech_list.size(1))
                            speech_list = speech_list.flatten(0, 1)
                            speech_list = to_device(
                                speech_list, "cuda" if ngpu > 0 else "cpu"
                            )
                            if task_token is None:
                                task_tokens = None
                            else:
                                task_tokens = to_device(
                                    task_token.repeat(speech_list.size(0)),
                                    "cuda" if ngpu > 0 else "cpu",
                                ).unsqueeze(1)
                            spk_embds = model(
                                speech=speech_list,
                                spk_labels=None,
                                extract_embd=True,
                                task_tokens=task_tokens,
                            )
                            # removed to be use magnitude in qmf
                            # spk_embds = F.normalize(spk_embds, p=2, dim=1)
                            spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                            for uid, _spk_embd in zip(utt_id_list, spk_embds):
                                if average:
                                    spk_embd_dic[uid] = (
                                        _spk_embd.mean(0).detach().cpu().numpy()
                                    )
                                else:
                                    spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()

                            utt_id_list = []
                            speech_list = []

                        idx += 1

            if len(utt_id_list) != 0:
                speech_list = torch.stack(speech_list, dim=0)
                org_shape = (speech_list.size(0), speech_list.size(1))
                speech_list = speech_list.flatten(0, 1)
                speech_list = to_device(speech_list, "cuda" if ngpu > 0 else "cpu")
                if task_token is None:
                    task_tokens = None
                else:
                    task_tokens = to_device(
                        task_token.repeat(speech_list.size(0)),
                        "cuda" if ngpu > 0 else "cpu",
                    ).unsqueeze(1)
                spk_embds = model(
                    speech=speech_list,
                    spk_labels=None,
                    extract_embd=True,
                    task_tokens=task_tokens,
                )
                spk_embds = F.normalize(spk_embds, p=2, dim=1)
                spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                for uid, _spk_embd in zip(utt_id_list, spk_embds):
                    if average:
                        spk_embd_dic[uid] = _spk_embd.mean(0).detach().cpu().numpy()
                    else:
                        spk_embd_dic[uid] = _spk_embd.detach().cpu().numpy()

        else:  # embed_avg
            for utt_id, batch in iterator:
                utt_id_list = []
                speech_list = []
                if "task_tokens" in batch:
                    task_token = batch["task_tokens"][0]

                assert isinstance(batch, dict), type(batch)
                for _utt_id, _speech, _speech2, _speech3, _speech4 in zip(
                    utt_id,
                    batch["speech"],
                    batch["speech2"],
                    batch["speech3"],
                    batch["speech4"],
                ):
                    _utt_id_1, _utt_id_2 = _utt_id.split("*")
                    # speech, speech2 and speech3 are enrollment utterances for the speaker whose
                    # speakerid is _utt_id_1 and speech4 is the test utterance
                    # for the purpose of enrollment, the uttid for each enrollment utterance will be
                    # recorded as _utt_id_1-first, _utt_id_1-second, and _utt_id_1-third
                    if (_utt_id_1 + "-first") not in utt_id_list:
                        utt_id_list.append(_utt_id_1 + "-first")
                        speech_list.append(
                            to_device(_speech, "cuda" if ngpu > 0 else "cpu")
                        )
                    if (_utt_id_1 + "-second") not in utt_id_list:
                        utt_id_list.append(_utt_id_1 + "-second")
                        speech_list.append(
                            to_device(_speech2, "cuda" if ngpu > 0 else "cpu")
                        )
                    if (_utt_id_1 + "-third") not in utt_id_list:
                        utt_id_list.append(_utt_id_1 + "-third")
                        speech_list.append(
                            to_device(_speech3, "cuda" if ngpu > 0 else "cpu")
                        )
                    if _utt_id_2 not in utt_id_list:
                        utt_id_list.append(_utt_id_2)
                        speech_list.append(
                            to_device(_speech4, "cuda" if ngpu > 0 else "cpu")
                        )

                assert len(utt_id_list) == len(speech_list)

                # extract speaker embeddings.
                n_utt = len(utt_id_list)
                for ii in range(0, n_utt, custom_bs):
                    _utt_ids = utt_id_list[ii : ii + custom_bs]
                    _speechs = speech_list[ii : ii + custom_bs]
                    _speechs = torch.stack(_speechs, dim=0)
                    org_shape = (_speechs.size(0), _speechs.size(1))
                    _speechs = _speechs.flatten(0, 1)
                    _speechs = to_device(_speechs, "cuda" if ngpu > 0 else "cpu")

                    if task_token is None:
                        task_tokens = None
                    else:
                        task_tokens = to_device(
                            task_token.repeat(_speechs.size(0)),
                            "cuda" if ngpu > 0 else "cpu",
                        ).unsqueeze(1)
                    spk_embds = model(
                        speech=_speechs,
                        spk_labels=None,
                        extract_embd=True,
                        task_tokens=task_tokens,
                    )

                    spk_embds = F.normalize(spk_embds, p=2, dim=1)
                    spk_embds = spk_embds.view(org_shape[0], org_shape[1], -1)

                    for _utt_id, _spk_embd in zip(_utt_ids, spk_embds):
                        if (
                            _utt_id.endswith("-first")
                            or _utt_id.endswith("-second")
                            or _utt_id.endswith("-third")
                        ):
                            spkID = _utt_id.split("-")[0]
                            if spkID in spk_embd_dic:
                                spk_embd_dic[spkID] = torch.cat(
                                    (spk_embd_dic[spkID], _spk_embd), dim=0
                                )
                            else:  # first time we see this speaker
                                spk_embd_dic[spkID] = _spk_embd
                        else:  # test utterance
                            spk_embd_dic[_utt_id] = _spk_embd

                    del utt_id_list
                    del speech_list

        # Compute the average embedding for each speaker and to cpu
        for spkID in spk_embd_dic:
            spk_embd_dic[spkID] = (
                torch.mean(spk_embd_dic[spkID], dim=0, keepdim=True)
                .detach()
                .cpu()
                .numpy()
            )

        np.savez(output_dir + f"/embeddings{rank}", **spk_embd_dic)
