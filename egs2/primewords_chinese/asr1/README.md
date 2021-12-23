# Note
- Dataset: http://www.openslr.org/47/
- Please double check the data preparation stage when using this recipe in your own setting. Some processing might be inconsistent with other sources (if any). Currently we do ***not*** find a standard reference for data preparation, so the train/dev/test split is ***not*** "official". We do ***not*** include real speaker ids as well. Instead, utterance ids are used in `utt2spk`.


# Conformer + Speed Perturbation + SpecAugment, without LM

