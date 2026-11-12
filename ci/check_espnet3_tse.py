"""Compare packed TSE and demo output with the native trained backend."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from espnet2.bin.enh_tse_inference import SeparateSpeech
from espnet3.publication import InferenceModel
from espnet3.publication.demo.session import load_demo_session


def main():
    """Validate real audio artifacts, model packing and demo input binding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = args.recipe.resolve()
    sample = json.loads((recipe / "data/test.json").read_text())[0]
    exp = recipe / "exp/training"
    native = SeparateSpeech(
        train_config=exp / "config.yaml", model_file=exp / "last.ckpt"
    )
    mixture, rate = sf.read(sample["speech_mix"], dtype="float32")
    enrollment, _ = sf.read(sample["enroll_ref1"], dtype="float32")
    expected = native(mixture, fs=rate, enroll_ref1=enrollment)[0]
    packed = InferenceModel.from_packed(exp / "model_pack")
    result = packed({"speech_mix": (rate, mixture), "enroll_ref1": (rate, enrollment)})
    assert result["inf"].rate == rate
    np.testing.assert_allclose(result["inf"].array, expected, atol=1e-6)
    session = load_demo_session(recipe / "demo", recipe / "demo/demo.yaml")
    callback = session.create_inference_fn(session.input_specs, session.output_specs)
    out_rate, out_audio = callback((rate, mixture), (rate, enrollment))
    assert out_rate == rate
    np.testing.assert_allclose(out_audio, expected, atol=1e-6)
    scp = exp / "inference/test/inf.scp"
    uid, path = scp.read_text().strip().split(maxsplit=1)
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = recipe / artifact
    waveform, file_rate = sf.read(artifact)
    assert uid == sample["utt_id"] and file_rate == rate
    np.testing.assert_allclose(waveform, expected, atol=1e-4)
    assert (exp / "inference/metrics.json").is_file()


if __name__ == "__main__":
    main()
