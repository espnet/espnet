from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from espnet3.components.modeling.hf_models import (
    AbsHFInferenceWrapper,
    AbsHFTrainingWrapper,
)


class TestTrainingWrapper(AbsHFTrainingWrapper):
    def collect_feats(self, **batch):
        return batch


class TestInferenceWrapper(AbsHFInferenceWrapper):
    def forward(self, inputs):
        return self.model(inputs)


@pytest.fixture
def mocked_training_model():
    model = MagicMock()
    processor = MagicMock()

    with (
        patch.object(
            TestTrainingWrapper,
            "model_class",
            create=True,
        ) as model_class,
        patch.object(
            TestTrainingWrapper,
            "processor_class",
            create=True,
        ) as processor_class,
    ):
        model_class.from_pretrained.return_value = model
        processor_class.from_pretrained.return_value = processor
        yield model, processor, model_class, processor_class


@pytest.fixture
def mocked_inference_model():
    model = MagicMock()
    processor = MagicMock()

    with (
        patch.object(
            TestInferenceWrapper,
            "model_class",
            create=True,
        ) as model_class,
        patch.object(
            TestInferenceWrapper,
            "processor_class",
            create=True,
        ) as processor_class,
    ):
        model_class.from_pretrained.return_value = model
        processor_class.from_pretrained.return_value = processor
        yield model, processor, model_class, processor_class


def test_training_wrapper_init(mocked_training_model):
    model, processor, model_class, processor_class = mocked_training_model

    wrapper = TestTrainingWrapper("test-model")

    model_class.from_pretrained.assert_called_once_with("test-model")
    processor_class.from_pretrained.assert_called_once_with("test-model")

    assert wrapper.model is model
    assert wrapper.processor is processor


def test_inference_wrapper_init(mocked_inference_model):
    model, processor, model_class, processor_class = mocked_inference_model

    wrapper = TestInferenceWrapper("test-model")

    model_class.from_pretrained.assert_called_once_with("test-model")
    processor_class.from_pretrained.assert_called_once_with("test-model")

    assert wrapper.model is model
    assert wrapper.processor is processor


def test_training_wrapper_forward(mocked_training_model):
    model, _, _, _ = mocked_training_model
    wrapper = TestTrainingWrapper("test-model")

    loss = torch.tensor(2.5, requires_grad=True)
    logits = torch.randn(4, 10)

    model.return_value = SimpleNamespace(
        loss=loss,
        logits=logits,
    )

    wrapper(
        input_ids=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
    )

    model.assert_called_once()


def test_training_wrapper_collect_feats():
    with (
        patch.object(
            TestTrainingWrapper.model_class,
            "from_pretrained",
            return_value=MagicMock(),
        ),
        patch.object(
            TestTrainingWrapper.processor_class,
            "from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        wrapper = TestTrainingWrapper("test-model")

    batch = {
        "input_values": torch.randn(2, 100),
        "input_lengths": torch.tensor([100, 80]),
    }

    assert wrapper.collect_feats(**batch) == batch


def test_training_wrapper_save_pretrained(mocked_training_model, tmp_path):
    model, processor, _, _ = mocked_training_model
    wrapper = TestTrainingWrapper("test-model")

    wrapper.save_pretrained(tmp_path)

    model.save_pretrained.assert_called_once_with(tmp_path)
    processor.save_pretrained.assert_called_once_with(tmp_path)


def test_inference_wrapper_forward(mocked_inference_model):
    model, _, _, _ = mocked_inference_model
    wrapper = TestInferenceWrapper("test-model")

    expected_output = MagicMock()
    model.return_value = expected_output

    inputs = {"input_values": torch.randn(2, 100)}

    output = wrapper(inputs)

    model.assert_called_once_with(inputs)
    assert output is expected_output
