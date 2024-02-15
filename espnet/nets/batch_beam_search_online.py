"""Parallel beam search module for online simulation."""

import logging
from typing import Any  # noqa: H301
from typing import Dict  # noqa: H301
from typing import List  # noqa: H301
from typing import Tuple  # noqa: H301

import torch

from espnet2.asr.transducer.beam_search_transducer_streaming import (
    BeamSearchTransducerStreaming,
)
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: H301
from espnet.nets.batch_beam_search import BatchHypothesis  # noqa: H301
from espnet.nets.beam_search import Hypothesis
from espnet.nets.beam_search_timesync_streaming import BeamSearchTimeSyncStreaming
from espnet.nets.e2e_asr_common import end_detect


class BatchBeamSearchOnline(BatchBeamSearch):
    """Online beam search implementation.

    This simulates streaming decoding.
    It requires encoded features of entire utterance and
    extracts block by block from it as it shoud be done
    in streaming processing.
    This is based on Tsunoo et al, "STREAMING TRANSFORMER ASR
    WITH BLOCKWISE SYNCHRONOUS BEAM SEARCH"
    (https://arxiv.org/abs/2006.14941).
    """

    def __init__(
        self,
        *args,
        block_size=40,
        hop_size=16,
        look_ahead=16,
        disable_repetition_detection=False,
        encoded_feat_length_limit=0,
        decoder_text_length_limit=0,
        incremental_decode=False,
        time_sync=False,
        ctc=None,
        hold_n=0,
        transducer_conf=None,
        joint_network=None,
        **kwargs,
    ):
        """Initialize beam search."""
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.disable_repetition_detection = disable_repetition_detection
        self.encoded_feat_length_limit = encoded_feat_length_limit
        self.decoder_text_length_limit = decoder_text_length_limit
        self.incremental_decode = incremental_decode
        self.time_sync = time_sync
        self.ctc = ctc
        self.hold_n = hold_n

        if time_sync:
            if transducer_conf is not None:
                self.time_sync_search = BeamSearchTransducerStreaming(
                    decoder=self.scorers["decoder"],
                    joint_network=joint_network,
                    beam_size=self.beam_size,
                    token_list=self.token_list,
                    hold_n=hold_n,
                    **transducer_conf,
                )
                del self.scorers["decoder"]
                self.t = 0
            else:
                scorers = self.scorers.copy()
                scorers["ctc"] = ctc
                self.time_sync_search = BeamSearchTimeSyncStreaming(
                    beam_size=self.beam_size,
                    weights=self.weights,
                    scorers=scorers,
                    sos=self.sos,
                    token_list=self.token_list,
                    hold_n=hold_n,
                )
                self.t = 0

        self.reset()

    def reset(self):
        """Reset parameters."""
        self.encbuffer = None
        self.running_hyps = None
        self.prev_hyps = []
        self.ended_hyps = []
        self.processed_block = 0
        self.process_idx = 0
        self.prev_output = None
        self.prev_incremental = None

    def score_full(
        self,
        hyp: BatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
            pre_x (torch.Tensor): Encoded speech feature for
                sequential attention (T, D)

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if (
                self.decoder_text_length_limit > 0
                and len(hyp.yseq) > 0
                and len(hyp.yseq[0]) > self.decoder_text_length_limit
            ):
                temp_yseq = hyp.yseq.narrow(
                    1, -self.decoder_text_length_limit, self.decoder_text_length_limit
                ).clone()
                temp_yseq[:, 0] = self.sos
                self.running_hyps.states["decoder"] = [
                    None for _ in self.running_hyps.states["decoder"]
                ]
            else:
                temp_yseq = hyp.yseq

            if "decoder" in k and self.return_hs:
                (scores[k], hs), states[k] = d.batch_score(
                    temp_yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            elif "decoder" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x, pre_x)
            else:
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x)

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        if self.encbuffer is None or self.block_size == 0:
            self.encbuffer = x
        else:
            self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

        x = self.encbuffer

        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * x.size(0))

        # set block_size == 0 for recomputing
        if self.block_size == 0:
            block_is_final = is_final
            h = x
            logging.debug(
                "  Feature length: {}, current position: {}".format(
                    h.shape[0], self.process_idx
                )
            )

            if self.running_hyps is None:  # init hyps
                init_scores = {}
                init_states = {}
                if "ctc" in self.scorers.keys():
                    self.scorers["ctc"].batch_init_state(h)
                    init_scores["ctc"] = torch.tensor([0.0])
                    init_states["ctc"] = [None]
                if "length_bonus" in self.scorers.keys():
                    init_scores["length_bonus"] = torch.tensor([0.0])
                    init_states["length_bonus"] = [None]
                if "decoder" in self.scorers.keys():
                    init_scores["decoder"] = torch.tensor([0.0])
                    init_states["decoder"] = [None]
                self.running_hyps = BatchHypothesis(
                    score=torch.tensor([0.0]),
                    scores=init_scores,
                    states=init_states,
                    length=torch.tensor([2]),
                    yseq=torch.tensor([self.hyp_primer], device=x.device),
                    hs=[],
                )
                self.prev_incremental = self.running_hyps

            if self.time_sync:
                ret = self.process_one_block_time_sync(
                    h, block_is_final, maxlen, maxlenratio
                )
            else:
                ret = self.process_one_block(
                    h, block_is_final, maxlen - self.process_idx, maxlenratio
                )
            logging.debug("Finished processing chunk: %d", self.processed_block)
            self.processed_block += 1

            # prune running_hyps, taking top as an incremental decoding
            if self.incremental_decode and not is_final:
                if (
                    self.running_hyps.yseq.shape[0] == 0
                ):  # running_hyps will be empty if maxlen is reached
                    logging.info(
                        "search stopped by maxlen in a non final chunk. \
                        reverting to prev running hyp"
                    )
                    self.running_hyps = self.prev_incremental
                logging.info(
                    "Hyps before incremental pruning: %d",
                    self.running_hyps.yseq.shape[0],
                )
                if self.running_hyps.yseq.shape[0] > 0:
                    self.running_hyps = self._batch_select(self.running_hyps, [0])
                    self.prev_incremental = self.running_hyps
                logging.info(
                    "Hyps after incremental pruning: %d",
                    self.running_hyps.yseq.shape[0],
                )

                if self.token_list is not None:
                    logging.info(
                        "best running hypo: "
                        + "".join(
                            [self.token_list[x] for x in self.running_hyps.yseq[0, 1:]]
                        )
                    )

                # hold_n
                if self.hold_n > 0 and self.running_hyps.length[0] > 2:
                    self.running_hyps = BatchHypothesis(
                        score=self.running_hyps.score,
                        scores=self.running_hyps.scores,
                        states=self.running_hyps.states,
                        length=self.running_hyps.length - self.hold_n,
                        yseq=self.running_hyps.yseq[:, : -self.hold_n],
                        hs=[],
                    )
                    if self.token_list is not None:
                        logging.info(
                            "best hypo after hold: "
                            + "".join(
                                [
                                    self.token_list[x]
                                    for x in self.running_hyps.yseq[0, 1:]
                                ]
                            )
                        )

            if is_final:
                if len(ret) == 0:
                    if self.prev_output is None:
                        return []
                    else:
                        return self.prev_output
                else:
                    return ret
            else:
                # dont return incremental hyps,
                # check them by grabbing top running_hyp
                if len(ret) > 0:
                    self.prev_output = ret
                return []

        # blockwise processing w/ rewinding
        else:
            ret = None
            while True:
                cur_end_frame = (
                    self.block_size
                    - self.look_ahead
                    + self.hop_size * self.processed_block
                )
                if cur_end_frame < x.shape[0]:
                    h = x.narrow(0, 0, cur_end_frame)
                    block_is_final = False
                else:
                    if is_final:
                        h = x
                        block_is_final = True
                    else:
                        break

                logging.debug("Start processing block: %d", self.processed_block)
                logging.debug(
                    "  Feature length: {}, current position: {}".format(
                        h.shape[0], self.process_idx
                    )
                )
                if (
                    self.encoded_feat_length_limit > 0
                    and h.shape[0] > self.encoded_feat_length_limit
                ):
                    h = h.narrow(
                        0,
                        h.shape[0] - self.encoded_feat_length_limit,
                        self.encoded_feat_length_limit,
                    )

                if self.running_hyps is None:
                    self.running_hyps = self.init_hyp(h)
                if self.time_sync:
                    ret = self.process_one_block_time_sync(
                        h, block_is_final, maxlen, maxlenratio
                    )
                else:
                    ret = self.process_one_block(
                        h, block_is_final, maxlen, minlen, maxlenratio
                    )
                logging.debug("Finished processing block: %d", self.processed_block)
                self.processed_block += 1

                # prune running_hyps, taking top as an incremental decoding
                if self.incremental_decode:
                    logging.debug(
                        "Hyps before incremental pruning: %d",
                        self.running_hyps.yseq.shape[0],
                    )
                    if self.running_hyps.yseq.shape[0] > 0:
                        self.running_hyps = self._batch_select(self.running_hyps, [0])
                    logging.debug(
                        "Hyps after incremental pruning: %d",
                        self.running_hyps.yseq.shape[0],
                    )

                if block_is_final:
                    return ret
            if ret is None:
                if self.prev_output is None:
                    return []
                else:
                    return self.prev_output
            else:
                self.prev_output = ret
                # N-best results
                return ret

    def process_one_block_time_sync(self, h, is_final, maxlen, maxlenratio):
        """Recognize one block w/ time sync."""
        hyps = self.time_sync_search(
            h,
            start_idx=self.t,
            is_final=is_final,
            incremental_decode=self.incremental_decode,
        )
        logging.debug("time:" + str(self.t))
        logging.debug("best_hyp:" + "".join([self.token_list[x] for x in hyps[0].yseq]))
        if is_final:
            self.t = 0
        else:
            self.t = len(h)
        return hyps

    def process_one_block(self, h, is_final, maxlen, minlen, maxlenratio):
        """Recognize one block."""
        # extend states for ctc
        self.extend(h, self.running_hyps)
        while self.process_idx < maxlen:
            logging.debug("position " + str(self.process_idx))
            best = self.search(self.running_hyps, h)

            if self.process_idx == maxlen - 1:
                # end decoding
                self.running_hyps = self.post_process(
                    self.process_idx, maxlen, minlen, maxlenratio, best, self.ended_hyps
                )
            n_batch = best.yseq.shape[0]
            local_ended_hyps = []
            is_local_eos = best.yseq[torch.arange(n_batch), best.length - 1] == self.eos
            prev_repeat = False
            for i in range(is_local_eos.shape[0]):
                if is_local_eos[i]:
                    hyp = self._select(best, i)
                    local_ended_hyps.append(hyp)
                # NOTE(tsunoo): check repetitions here
                # This is a implicit implementation of
                # Eq (11) in https://arxiv.org/abs/2006.14941
                # A flag prev_repeat is used instead of using set
                # NOTE(fujihara): I made it possible to turned off
                # the below lines using disable_repetition_detection flag,
                # because this criteria is too sensitive that the beam
                # search starts only after the entire inputs are available.
                # Empirically, this flag didn't affect the performance.
                elif (
                    not self.disable_repetition_detection
                    and not prev_repeat
                    and best.yseq[i, -1] in best.yseq[i, :-1]
                    and not is_final
                ):
                    prev_repeat = True
            if prev_repeat:
                logging.info("Detected repetition.")
                break

            if (
                is_final
                and maxlenratio == 0.0
                and end_detect(
                    [lh.asdict() for lh in self.ended_hyps], self.process_idx
                )
            ):
                logging.info(f"end detected at {self.process_idx}")
                return self.assemble_hyps(self.ended_hyps)

            if len(local_ended_hyps) > 0 and not is_final:
                logging.info("Detected hyp(s) reaching EOS in this block.")
                break
                # breaking here means that prev hyps
                # is 2 behind the ended hyps, not 1

            self.prev_hyps = self.running_hyps
            self.running_hyps = self.post_process(
                self.process_idx, maxlen, minlen, maxlenratio, best, self.ended_hyps
            )

            if is_final:
                for hyp in local_ended_hyps:
                    self.ended_hyps.append(hyp)

            if len(self.running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                return self.assemble_hyps(self.ended_hyps)
            else:
                logging.debug(f"remained hypotheses: {len(self.running_hyps)}")
            # increment number
            self.process_idx += 1

        if is_final:
            return self.assemble_hyps(self.ended_hyps)
        else:
            try:
                local_ended_hyps
            except Exception:
                local_ended_hyps = []

            for hyp in self.ended_hyps:
                local_ended_hyps.append(hyp)
            rets = self.assemble_hyps(local_ended_hyps)

            if self.process_idx > 1 and len(self.prev_hyps) > 0:
                self.running_hyps = self.prev_hyps
                self.process_idx -= 1
                self.prev_hyps = []

            # N-best results
            return rets

    def assemble_hyps(self, ended_hyps):
        """Assemble the hypotheses."""
        if self.normalize_length:
            # Note (Jinchuan): -1 since hyp starts with <sos> and
            # initially has score of 0.0
            nbest_hyps = sorted(
                ended_hyps, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return []

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def extend(self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The extended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
