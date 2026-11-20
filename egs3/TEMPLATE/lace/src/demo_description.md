Upload a recording or speak into the microphone, then press **Run**. The audio is
resampled to the codec's rate and averaged to one channel, encoded with the
packed LACE model at its compression rate (every codebook layer merges similar
neighbouring frames into segments and runs at its own, lower frame rate), and
decoded back; what you hear is the codec's resynthesis of your input. Clean
speech of a few seconds gives the clearest comparison.
