"""Contract imports and failures across actual process worker boundaries."""

import sys

import pytest
from omegaconf import OmegaConf

from espnet3.contract.kinds.text import TextKind
from espnet3.systems.base.inference import infer


@pytest.mark.execution_timeout(90)
@pytest.mark.parametrize("fail", [False, True])
def test_worker_imports_custom_kind_and_propagates_failure(tmp_path, monkeypatch, fail):
    import espnet3.parallel.parallel as parallel_module

    monkeypatch.setattr(parallel_module, "parallel_config", None)
    module_name = "inference_worker_contract_fixture"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / f"{module_name}.py").write_text(
        """from espnet3.contract import register_kind
from espnet3.contract.kinds.text import TextKind
from espnet3.contract.normalizers import register_normalizer
from espnet3.systems.base.inference_provider import InferenceProvider

register_kind("worker_text", TextKind(), replace=True)
register_normalizer("worker_suffix", lambda text: text + "!", replace=True)

class Provider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        return [{"text": "a", "utt_id": "a"}, {"text": "b", "utt_id": "b"}]

    @staticmethod
    def build_model(config):
        def model(text):
            if config.fail:
                raise RuntimeError("worker backend failed")
            return {"text": text}
        return model
""",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "fail": fail,
            "model": {},
            "dataset": {"test": [{"name": "test"}]},
            "parallel": {
                "env": "local",
                "n_workers": 2,
                "options": {
                    "processes": True,
                    "threads_per_worker": 1,
                    "dashboard_address": None,
                },
            },
            "inference_dir": str(tmp_path / "decode"),
            "input_key": "text",
            "batch_size": 1,
            "provider": {"_target_": f"{module_name}.Provider"},
            "runner": {
                "_target_": "espnet3.systems.base.inference_runner.InferenceRunner"
            },
            "contract": {
                "imports": [module_name],
                "inputs": {
                    "text": {"kind": "worker_text", "normalize": ["worker_suffix"]}
                },
                "outputs": {
                    "text": {"kind": "worker_text", "normalize": ["worker_suffix"]}
                },
            },
        }
    )
    # Restore the parent's registries as well as the temporary import path.
    from espnet3.contract import KINDS
    from espnet3.contract.normalizers import NORMALIZERS

    monkeypatch.setitem(KINDS, "worker_text", TextKind())
    monkeypatch.setitem(NORMALIZERS, "worker_suffix", lambda text: text)
    output = tmp_path / "decode" / "test"
    if fail:
        with pytest.raises(RuntimeError, match="worker backend failed"):
            infer(config)
        assert not (output / "text.scp").exists()
        assert not list(output.glob("split.*/done"))
    else:
        infer(config)
        assert (output / "text.scp").read_text().splitlines() == ["a a!!", "b b!!"]
        assert len(list(output.glob("split.*/done"))) == 2
    sys.modules.pop(module_name, None)
