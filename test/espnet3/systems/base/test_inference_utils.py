import pytest

from espnet3.systems.base.inference_runner import _materialize_output_value


def test_materialize_output_value_rejects_top_level_lists(tmp_path):
    with pytest.raises(TypeError, match="Top-level list outputs are not supported"):
        _materialize_output_value(
            idx_value="utt1",
            field_key="hyp",
            value=["h1", "h2"],
            output_dir=tmp_path,
            artifact_config=None,
        )
