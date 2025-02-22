"""Streaming/window module."""

import torch


# TODO(pzelasko): Currently allows half-streaming only;
#  needs streaming attention decoder implementation
class WindowStreamingE2E(object):
    """WindowStreamingE2E constructor.

    :param E2E e2e: E2E ASR object
    :param recog_args: arguments for "recognize" method of E2E
    """

    def __init__(self, e2e, recog_args, rnnlm=None):
        """Initialize WindowStreaming."""
        self._e2e = e2e
        self._recog_args = recog_args
        self._char_list = e2e.char_list
        self._rnnlm = rnnlm

        self._e2e.eval()

        self._offset = 0
        self._previous_encoder_recurrent_state = None
        self._encoder_states = []
        self._ctc_posteriors = []
        self._last_recognition = None

        assert (
            self._recog_args.ctc_weight > 0.0
        ), "WindowStreamingE2E works only with combined CTC and attention decoders."

    def accept_input(self, x):
        """Call this method each time a new batch of input is available."""
        h, ilen = self._e2e.subsample_frames(x)

        # Streaming encoder
        h, _, self._previous_encoder_recurrent_state = self._e2e.enc(
            h.unsqueeze(0), ilen, self._previous_encoder_recurrent_state
        )
        self._encoder_states.append(h.squeeze(0))

        # CTC posteriors for the incoming audio
        self._ctc_posteriors.append(self._e2e.ctc.log_softmax(h).squeeze(0))

    def _input_window_for_decoder(self, use_all=False):
        """Generate input window for decoder."""
        if use_all:
            return (
                torch.cat(self._encoder_states, dim=0),
                torch.cat(self._ctc_posteriors, dim=0),
            )

        def select_unprocessed_windows(window_tensors):
            last_offset = self._offset
            offset_traversed = 0
            selected_windows = []
            for es in window_tensors:
                if offset_traversed > last_offset:
                    selected_windows.append(es)
                    continue
                offset_traversed += es.size(1)
            return torch.cat(selected_windows, dim=0)

        return (
            select_unprocessed_windows(self._encoder_states),
            select_unprocessed_windows(self._ctc_posteriors),
        )

    def decode_with_attention_offline(self):
        """Run the attention decoder offline.

        Works even if the previous layers (encoder and CTC decoder) were
        being run in the online mode.
        This method should be run after all the audio has been consumed.
        This is used mostly to compare the results between offline
        and online implementation of the previous layers.
        """
        h, lpz = self._input_window_for_decoder(use_all=True)

        return self._e2e.dec.recognize_beam(
            h, lpz, self._recog_args, self._char_list, self._rnnlm
        )
