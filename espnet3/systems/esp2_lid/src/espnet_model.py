"""ESPnet2 LID model with sequence lengths for ESPnet3 statistics."""

from espnet2.lid.espnet_model import ESPnetLIDModel as ESPnet2LIDModel


class ESPnetLIDModel(ESPnet2LIDModel):
    """Preserve LID training and inference while exposing feature lengths."""

    def collect_feats(self, speech, speech_lengths, **kwargs):
        """Exclude padding and accumulate statistics over feature frames."""
        feats, lengths = self.extract_feats(speech, speech_lengths)
        return {
            "feats": feats,
            "feats_lengths": speech_lengths if lengths is None else lengths,
        }
