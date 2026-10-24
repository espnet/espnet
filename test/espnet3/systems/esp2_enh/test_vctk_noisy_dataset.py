import numpy as np
import soundfile as sf

from egs3.vctk_noisy.esp2_enh.dataset.dataset import VCTKNoisyDataset


def test_vctk_noisy_dataset_resamples_48k_corpus_to_16k(tmp_path):
    # The official VCTK-DEMAND wavs are 48 kHz; the recipe trains at 16 kHz.
    for folder in ("noisy_testset_wav", "clean_testset_wav"):
        (tmp_path / folder).mkdir()
        sf.write(tmp_path / folder / "p232_001.wav", np.zeros(48000), 48000)

    dataset = VCTKNoisyDataset("test", data_path=tmp_path, inference=True)
    item = dataset[0]

    assert item["speech_mix"].shape == (16000,)
    assert item["speech_ref1"].shape == (16000,)
