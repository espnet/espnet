import pytest

from espnet3.systems.base.inference_runner import InferenceRunner, _load_output_fn


class DummyProvider:
    pass


class DummyRunner(InferenceRunner):
    @staticmethod
    def forward(idx, *, dataset, model, **env):
        return {"idx": idx, "hyp": "h", "ref": "r"}


def test_resolve_idx_key_uses_default_utt_id():
    runner = DummyRunner(DummyProvider())

    assert runner.resolve_idx_key({"utt_id": "u1", "hyp": "h", "ref": "r"}) == "utt_id"


def test_resolve_idx_key_uses_explicit_configured_key():
    runner = DummyRunner(DummyProvider(), idx_key="sample_id")

    assert (
        runner.resolve_idx_key({"sample_id": "u1", "hyp": "h", "ref": "r"})
        == "sample_id"
    )


def test_resolve_idx_key_rejects_missing_configured_key():
    runner = DummyRunner(DummyProvider(), idx_key="sample_id")

    with pytest.raises(ValueError, match="idx_key='sample_id'"):
        runner.resolve_idx_key({"utt_id": "u1", "hyp": "h", "ref": "r"})


def test_load_output_fn_rejects_missing_module():
    with pytest.raises(ModuleNotFoundError):
        _load_output_fn("no.such.module.output_fn")


def test_forward_raises_without_input_key_kwarg():
    with pytest.raises(RuntimeError, match="input_key must be provided"):
        InferenceRunner.forward(0, dataset=[], model=lambda: None)


def test_forward_single_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            0, dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_raises_key_error_for_missing_dataset_key():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]
    with pytest.raises(KeyError, match="Input key"):
        InferenceRunner.forward(
            [0, 1], dataset=dataset, model=lambda **kw: None, input_key="text"
        )


def test_forward_batched_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        [0, 1], dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": [1.0, 2.0]}


def test_forward_single_returns_model_output_without_output_fn():
    dataset = [{"speech": 1.0}]

    def model(speech):
        return {"result": speech}

    result = InferenceRunner.forward(
        0, dataset=dataset, model=model, input_key="speech"
    )
    assert result == {"result": 1.0}


def test_forward_single_passes_model_kwargs_to_model():
    dataset = [{"speech": 1.0}]

    def model(speech, beam_size):
        return {"result": f"{speech}:{beam_size}"}

    result = InferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key="speech",
        model_kwargs={"beam_size": 2},
    )
    assert result == {"result": "1.0:2"}


def test_forward_batched_passes_model_kwargs_to_model():
    dataset = [{"speech": 1.0}, {"speech": 2.0}]

    def model(speech, beam_size):
        return {"result": f"{speech}:{beam_size}"}

    result = InferenceRunner.forward(
        [0, 1],
        dataset=dataset,
        model=model,
        input_key="speech",
        model_kwargs={"beam_size": 4},
    )
    assert result == {"result": "[1.0, 2.0]:4"}


def test_forward_batched_wraps_model_exception_in_runtime_error():
    dataset = [{"speech": 1.0}]

    def failing_model(speech):
        raise ValueError("model broken")

    with pytest.raises(RuntimeError, match="Batched inference failed"):
        InferenceRunner.forward(
            [0], dataset=dataset, model=failing_model, input_key="speech"
        )
