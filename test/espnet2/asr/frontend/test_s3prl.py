from distutils.version import LooseVersion
import os

import torch

is_torch_1_7_plus = LooseVersion(torch.__version__) >= LooseVersion("1.7.0")

if is_torch_1_7_plus:
    from s3prl.upstream.interfaces import Featurizer


def test_frontend_output_size():
    # Skip some testing cases
    if not is_torch_1_7_plus:
        return

    s3prl_path = None
    python_path_list = os.environ.get("PYTHONPATH", "(None)").split(":")
    for p in python_path_list:
        if p.endswith("s3prl"):
            s3prl_path = p
            break
    assert s3prl_path is not None

    s3prl_upstream = torch.hub.load(
        s3prl_path,
        "mel",
        source="local",
    ).to("cpu")

    feature_selection = "last_hidden_state"
    s3prl_featurizer = Featurizer(
        upstream=s3prl_upstream,
        feature_selection=feature_selection,
        upstream_device="cpu",
    )

    wavs = [torch.randn(1600)]
    feats = s3prl_upstream(wavs)
    feats = s3prl_featurizer(wavs, feats)
    assert feats[0].shape[-1] == 80
