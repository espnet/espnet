from argparse import ArgumentParser
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from espnet2.bin.ser_inference import Speech2Emotion, get_parser, inference, main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


def test_speech2emotion_init():
    """Test Speech2Emotion initialization with mocked dependencies."""
    with patch("espnet2.bin.ser_inference.SERTask") as mock_task:
        mock_model = MagicMock()
        mock_train_args = MagicMock()
        mock_task.build_model_from_file.return_value = (mock_model, mock_train_args)

        speech2emotion = Speech2Emotion(
            ser_train_config="dummy_config.yaml",
            ser_model_file="dummy_model.pth",
            device="cpu",
            batch_size=1,
            dtype="float32",
        )

        assert speech2emotion.ser_model is not None
        assert speech2emotion.device == "cpu"
        assert speech2emotion.dtype == "float32"


def test_speech2emotion_call_with_numpy():
    """Test Speech2Emotion __call__ with numpy input."""
    with patch("espnet2.bin.ser_inference.SERTask") as mock_task:
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0])
        mock_train_args = MagicMock()
        mock_task.build_model_from_file.return_value = (mock_model, mock_train_args)

        speech2emotion = Speech2Emotion(
            ser_train_config="dummy_config.yaml",
            ser_model_file="dummy_model.pth",
            device="cpu",
            dtype="float32",
        )

        # Test with numpy array input
        speech = np.random.randn(16000)
        result = speech2emotion(speech)

        assert result is not None
        mock_model.assert_called_once()


def test_speech2emotion_call_with_tensor():
    """Test Speech2Emotion __call__ with torch tensor input."""
    with patch("espnet2.bin.ser_inference.SERTask") as mock_task:
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0])
        mock_train_args = MagicMock()
        mock_task.build_model_from_file.return_value = (mock_model, mock_train_args)

        speech2emotion = Speech2Emotion(
            ser_train_config="dummy_config.yaml",
            ser_model_file="dummy_model.pth",
            device="cpu",
            dtype="float32",
        )

        # Test with torch tensor input
        speech = torch.randn(16000)
        result = speech2emotion(speech)

        assert result is not None
        mock_model.assert_called_once()


def test_speech2emotion_from_pretrained_without_model_tag():
    """Test Speech2Emotion.from_pretrained without model_tag."""
    with patch("espnet2.bin.ser_inference.SERTask") as mock_task:
        mock_model = MagicMock()
        mock_train_args = MagicMock()
        mock_task.build_model_from_file.return_value = (mock_model, mock_train_args)

        speech2emotion = Speech2Emotion.from_pretrained(
            model_tag=None,
            ser_train_config="dummy_config.yaml",
            ser_model_file="dummy_model.pth",
        )

        assert speech2emotion is not None


def test_speech2emotion_from_pretrained_with_model_tag():
    """Test Speech2Emotion.from_pretrained with model_tag."""
    with patch("espnet2.bin.ser_inference.SERTask") as mock_task, patch(
        "espnet2.bin.ser_inference.ModelDownloader"
    ) as mock_downloader:
        mock_model = MagicMock()
        mock_train_args = MagicMock()
        mock_task.build_model_from_file.return_value = (mock_model, mock_train_args)

        mock_d = MagicMock()
        mock_d.download_and_unpack.return_value = {
            "ser_train_config": "config.yaml",
            "ser_model_file": "model.pth",
        }
        mock_downloader.return_value = mock_d

        speech2emotion = Speech2Emotion.from_pretrained(model_tag="test_model")

        assert speech2emotion is not None
        mock_d.download_and_unpack.assert_called_once_with("test_model")


def test_inference_batch_size_error():
    """Test inference function raises error for batch_size > 1."""
    with pytest.raises(NotImplementedError):
        inference(
            output_dir="output",
            batch_size=2,
            dtype="float32",
            ngpu=0,
            seed=42,
            num_workers=1,
            log_level="INFO",
            data_path_and_name_and_type=[("dummy.scp", "speech", "sound")],
            key_file=None,
            ser_train_config="config.yaml",
            ser_model_file="model.pth",
            model_tag=None,
            allow_variable_data_keys=False,
        )


def test_inference_multi_gpu_error():
    """Test inference function raises error for ngpu > 1."""
    with pytest.raises(NotImplementedError):
        inference(
            output_dir="output",
            batch_size=1,
            dtype="float32",
            ngpu=2,
            seed=42,
            num_workers=1,
            log_level="INFO",
            data_path_and_name_and_type=[("dummy.scp", "speech", "sound")],
            key_file=None,
            ser_train_config="config.yaml",
            ser_model_file="model.pth",
            model_tag=None,
            allow_variable_data_keys=False,
        )


def test_inference_basic_flow(tmp_path):
    """Test basic inference flow with mocked dependencies."""
    output_dir = str(tmp_path / "output")

    with patch("espnet2.bin.ser_inference.Speech2Emotion") as mock_s2e_class, patch(
        "espnet2.bin.ser_inference.SERTask"
    ) as mock_task, patch("espnet2.bin.ser_inference.set_all_random_seed"), patch(
        "espnet2.bin.ser_inference.DatadirWriter"
    ) as mock_writer:

        # Mock Speech2Emotion instance
        mock_s2e = MagicMock()
        mock_s2e.ser_train_args = MagicMock()
        mock_s2e.return_value = torch.tensor([0])
        mock_s2e_class.from_pretrained.return_value = mock_s2e

        # Mock data loader
        mock_loader = [
            (
                ["key1"],
                {
                    "speech": torch.randn(1, 16000),
                    "speech_lengths": torch.tensor([16000]),
                },
            )
        ]
        mock_task.build_streaming_iterator.return_value = mock_loader
        mock_task.build_preprocess_fn.return_value = lambda x: x
        mock_task.build_collate_fn.return_value = lambda x: x

        # Mock writer
        mock_writer_instance = MagicMock()
        mock_writer.return_value.__enter__.return_value = mock_writer_instance

        # Run inference
        inference(
            output_dir=output_dir,
            batch_size=1,
            dtype="float32",
            ngpu=0,
            seed=42,
            num_workers=1,
            log_level="INFO",
            data_path_and_name_and_type=[("dummy.scp", "speech", "sound")],
            key_file=None,
            ser_train_config="config.yaml",
            ser_model_file="model.pth",
            model_tag=None,
            allow_variable_data_keys=False,
        )

        # Verify key components were called
        mock_s2e_class.from_pretrained.assert_called_once()
        mock_task.build_streaming_iterator.assert_called_once()
        mock_writer.assert_called_once()
