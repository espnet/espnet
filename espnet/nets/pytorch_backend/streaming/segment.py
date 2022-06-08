import numpy as np
import torch


class SegmentStreamingE2E(object):
    """SegmentStreamingE2E constructor.

    :param E2E e2e: E2E ASR object
    :param recog_args: arguments for "recognize" method of E2E
    """

    def __init__(self, e2e, recog_args, rnnlm=None):
        self._e2e = e2e
        self._recog_args = recog_args
        self._char_list = e2e.char_list
        self._rnnlm = rnnlm

        self._e2e.eval()

        self._blank_idx_in_char_list = -1
        for idx in range(len(self._char_list)):
            if self._char_list[idx] == self._e2e.blank:
                self._blank_idx_in_char_list = idx
                break

        self._subsampling_factor = np.prod(e2e.subsample)
        self._activates = 0
        self._blank_dur = 0

        self._previous_input = []
        self._previous_encoder_recurrent_state = None
        self._encoder_states = []
        self._ctc_posteriors = []

        assert (
            self._recog_args.batchsize <= 1
        ), "SegmentStreamingE2E works only with batch size <= 1"
        assert (
            "b" not in self._e2e.etype
        ), "SegmentStreamingE2E works only with uni-directional encoders"

    def accept_input(self, x):
        """Call this method each time a new batch of input is available."""

        self._previous_input.extend(x)
        h, ilen = self._e2e.subsample_frames(x)

        # Run encoder and apply greedy search on CTC softmax output
        h, _, self._previous_encoder_recurrent_state = self._e2e.enc(
            h.unsqueeze(0), ilen, self._previous_encoder_recurrent_state
        )
        z = self._e2e.ctc.argmax(h).squeeze(0)

        if self._activates == 0 and z[0] != self._blank_idx_in_char_list:
            self._activates = 1

            # Rerun encoder with zero state at onset of detection
            tail_len = self._subsampling_factor * (
                self._recog_args.streaming_onset_margin + 1
            )
            h, ilen = self._e2e.subsample_frames(
                np.reshape(
                    self._previous_input[-tail_len:], [-1, len(self._previous_input[0])]
                )
            )
            h, _, self._previous_encoder_recurrent_state = self._e2e.enc(
                h.unsqueeze(0), ilen, None
            )

        hyp = None
        if self._activates == 1:
            self._encoder_states.extend(h.squeeze(0))
            self._ctc_posteriors.extend(self._e2e.ctc.log_softmax(h).squeeze(0))

            if z[0] == self._blank_idx_in_char_list:
                self._blank_dur += 1
            else:
                self._blank_dur = 0

            if self._blank_dur >= self._recog_args.streaming_min_blank_dur:
                seg_len = (
                    len(self._encoder_states)
                    - self._blank_dur
                    + self._recog_args.streaming_offset_margin
                )
                if seg_len > 0:
                    # Run decoder with a detected segment
                    h = torch.cat(self._encoder_states[:seg_len], dim=0).view(
                        -1, self._encoder_states[0].size(0)
                    )
                    if self._recog_args.ctc_weight > 0.0:
                        lpz = torch.cat(self._ctc_posteriors[:seg_len], dim=0).view(
                            -1, self._ctc_posteriors[0].size(0)
                        )
                        if self._recog_args.batchsize > 0:
                            lpz = lpz.unsqueeze(0)
                        normalize_score = False
                    else:
                        lpz = None
                        normalize_score = True

                    if self._recog_args.batchsize == 0:
                        hyp = self._e2e.dec.recognize_beam(
                            h, lpz, self._recog_args, self._char_list, self._rnnlm
                        )
                    else:
                        hlens = torch.tensor([h.shape[0]])
                        hyp = self._e2e.dec.recognize_beam_batch(
                            h.unsqueeze(0),
                            hlens,
                            lpz,
                            self._recog_args,
                            self._char_list,
                            self._rnnlm,
                            normalize_score=normalize_score,
                        )[0]

                    self._activates = 0
                    self._blank_dur = 0

                    tail_len = (
                        self._subsampling_factor
                        * self._recog_args.streaming_onset_margin
                    )
                    self._previous_input = self._previous_input[-tail_len:]
                    self._encoder_states = []
                    self._ctc_posteriors = []

        return hyp
