import logging
import random
from argparse import Namespace

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six

import espnet.nets.chainer_backend.deterministic_embed_id as DL
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect

CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


class Decoder(chainer.Chain):
    """Decoder layer.

    Args:
        eprojs (int): Dimension of input variables from encoder.
        odim (int): The output dimension.
        dtype (str): Decoder type.
        dlayers (int): Number of layers for decoder.
        dunits (int): Dimension of input vector of decoder.
        sos (int): Number to indicate the start of sequences.
        eos (int): Number to indicate the end of sequences.
        att (Module): Attention module defined at
            `espnet.espnet.nets.chainer_backend.attentions`.
        verbose (int): Verbosity level.
        char_list (List[str]): List of all characters.
        labeldist (numpy.array): Distributed array of counted transcript length.
        lsm_weight (float): Weight to use when calculating the training loss.
        sampling_probability (float): Threshold for scheduled sampling.

    """

    def __init__(
        self,
        eprojs,
        odim,
        dtype,
        dlayers,
        dunits,
        sos,
        eos,
        att,
        verbose=0,
        char_list=None,
        labeldist=None,
        lsm_weight=0.0,
        sampling_probability=0.0,
    ):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed = DL.EmbedID(odim, dunits)
            self.rnn0 = (
                L.StatelessLSTM(dunits + eprojs, dunits)
                if dtype == "lstm"
                else L.StatelessGRU(dunits + eprojs, dunits)
            )
            for i in six.moves.range(1, dlayers):
                setattr(
                    self,
                    "rnn%d" % i,
                    L.StatelessLSTM(dunits, dunits)
                    if dtype == "lstm"
                    else L.StatelessGRU(dunits, dunits),
                )
            self.output = L.Linear(dunits, odim)
        self.dtype = dtype
        self.loss = None
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            c_list[0], z_list[0] = self.rnn0(c_prev[0], z_prev[0], ey)
            for i in six.moves.range(1, self.dlayers):
                c_list[i], z_list[i] = self["rnn%d" % i](
                    c_prev[i], z_prev[i], z_list[i - 1]
                )
        else:
            if z_prev[0] is None:
                xp = self.xp
                with chainer.backends.cuda.get_device_from_id(self._device_id):
                    z_prev[0] = chainer.Variable(
                        xp.zeros((ey.shape[0], self.dunits), dtype=ey.dtype)
                    )
            z_list[0] = self.rnn0(z_prev[0], ey)
            for i in six.moves.range(1, self.dlayers):
                if z_prev[i] is None:
                    xp = self.xp
                    with chainer.backends.cuda.get_device_from_id(self._device_id):
                        z_prev[i] = chainer.Variable(
                            xp.zeros(
                                (z_list[i - 1].shape[0], self.dunits),
                                dtype=z_list[i - 1].dtype,
                            )
                        )
                z_list[i] = self["rnn%d" % i](z_prev[i], z_list[i - 1])
        return z_list, c_list

    def __call__(self, hs, ys):
        """Core function of Decoder layer.

        Args:
            hs (list of chainer.Variable | N-dimension array):
                Input variable from encoder.
            ys (list of chainer.Variable | N-dimension array):
                Input variable of decoder.

        Returns:
            chainer.Variable: A variable holding a scalar array of the training loss.
            chainer.Variable: A variable holding a scalar array of the accuracy.

        """
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], "i")
        sos = self.xp.array([self.sos], "i")
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get dim, length info
        batch = pad_ys_out.shape[0]
        olength = pad_ys_out.shape[1]
        logging.info(
            self.__class__.__name__
            + " input lengths:  "
            + str(self.xp.array([h.shape[0] for h in hs]))
        )
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + str(self.xp.array([y.shape[0] for y in ys_out]))
        )

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(" scheduled sampling ")
                z_out = self.output(z_all[-1])
                z_out = F.argmax(F.log_softmax(z_out), axis=1)
                z_out = self.embed(z_out)
                ey = F.hstack((z_out, att_c))  # utt x (zdim + hdim)
            else:
                ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            z_all.append(z_list[-1])

        z_all = F.stack(z_all, axis=1).reshape(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.softmax_cross_entropy(y_all, F.flatten(pad_ys_out))
        # -1: eos, which is removed in the loss computation
        self.loss *= np.mean([len(x) for x in ys_in]) - 1
        acc = F.accuracy(y_all, F.flatten(pad_ys_out), ignore_label=-1)
        logging.info("att loss:" + str(self.loss.data))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = y_all.reshape(batch, olength, -1)
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
                idx_true = y_true_[y_true_ != -1]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat).replace("<space>", " ")
                seq_true = "".join(seq_true).replace("<space>", " ")
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = chainer.Variable(self.xp.asarray(self.labeldist))
            loss_reg = -F.sum(
                F.scale(F.log_softmax(y_all), self.vlabeldist, axis=1)
            ) / len(ys_in)
            self.loss = (1.0 - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        """Beam search implementation.

        Args:
            h (chainer.Variable): One of the output from the encoder.
            lpz (chainer.Variable | None): Result of net propagation.
            recog_args (Namespace): The argument.
            char_list (List[str]): List of all characters.
            rnnlm (Module): RNNLM module. Defined at `espnet.lm.chainer_backend.lm`

        Returns:
            List[Dict[str,Any]]: Result of recognition.

        """
        logging.info("input lengths: " + str(h.shape[0]))
        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        a = None
        self.att.reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        y = self.xp.full(1, self.sos, "i")
        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.shape[0]))
        minlen = int(recog_args.minlenratio * h.shape[0])
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {
                "score": 0.0,
                "yseq": [y],
                "c_prev": c_list,
                "z_prev": z_list,
                "a_prev": a,
                "rnnlm_prev": None,
            }
        else:
            hyp = {
                "score": 0.0,
                "yseq": [y],
                "c_prev": c_list,
                "z_prev": z_list,
                "a_prev": a,
            }
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz, 0, self.eos, self.xp)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                ey = self.embed(hyp["yseq"][i])  # utt list (1) x zdim
                att_c, att_w = self.att([h], hyp["z_prev"][0], hyp["a_prev"])
                ey = F.hstack((ey, att_c))  # utt(1) x (zdim + hdim)

                z_list, c_list = self.rnn_forward(
                    ey, z_list, c_list, hyp["z_prev"], hyp["c_prev"]
                )

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1])).data
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(
                        hyp["rnnlm_prev"], hyp["yseq"][i]
                    )
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][
                        :ctc_beam
                    ]
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids, hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids
                    ] + ctc_weight * (ctc_scores - hyp["ctc_score_prev"])
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids]
                        )
                    joint_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][
                        :beam
                    ]
                    local_best_scores = local_scores[:, joint_best_ids]
                    local_best_ids = local_best_ids[joint_best_ids]
                else:
                    local_best_ids = self.xp.argsort(local_scores, axis=1)[0, ::-1][
                        :beam
                    ]
                    local_best_scores = local_scores[:, local_best_ids]

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # do not copy {z,c}_list directly
                    new_hyp["z_prev"] = z_list[:]
                    new_hyp["c_prev"] = c_list[:]
                    new_hyp["a_prev"] = att_w
                    new_hyp["score"] = hyp["score"] + local_best_scores[0, j]
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = self.xp.full(
                        1, local_best_ids[j], "i"
                    )
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypotheses: " + str(len(hyps)))
            logging.debug(
                "best hypo: "
                + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]]).replace(
                    "<space>", " "
                )
            )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.xp.full(1, self.eos, "i"))

            # add ended hypotheses to a final list,
            # and removed them from current hypotheses
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remaining hypotheses: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            for hyp in hyps:
                logging.debug(
                    "hypo: "
                    + "".join([char_list[int(x)] for x in hyp["yseq"][1:]]).replace(
                        "<space>", " "
                    )
                )

            logging.debug("number of ended hypotheses: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, "
                "perform recognition again with smaller minlenratio."
            )
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )

        return nbest_hyps

    def calculate_all_attentions(self, hs, ys):
        """Calculate all of attentions.

        Args:
            hs (list of chainer.Variable | N-dimensional array):
                Input variable from encoder.
            ys (list of chainer.Variable | N-dimensional array):
                Input variable of decoder.

        Returns:
            chainer.Variable: List of attention weights.

        """
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], "i")
        sos = self.xp.array([self.sos], "i")
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get length info
        olength = pad_ys_out.shape[1]

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            att_ws.append(att_w)  # for debugging

        att_ws = F.stack(att_ws, axis=1)
        att_ws.to_cpu()

        return att_ws.data


def decoder_for(args, odim, sos, eos, att, labeldist):
    """Return the decoding layer corresponding to the args.

    Args:
        args (Namespace): The program arguments.
        odim (int): The output dimension.
        sos (int): Number to indicate the start of sequences.
        eos (int) Number to indicate the end of sequences.
        att (Module):
            Attention module defined at `espnet.nets.chainer_backend.attentions`.
        labeldist (numpy.array): Distributed array of length od transcript.

    Returns:
        chainer.Chain: The decoder module.

    """
    return Decoder(
        args.eprojs,
        odim,
        args.dtype,
        args.dlayers,
        args.dunits,
        sos,
        eos,
        att,
        args.verbose,
        args.char_list,
        labeldist,
        args.lsm_weight,
        args.sampling_probability,
    )
