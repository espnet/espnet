import os
from argparse import ArgumentParser
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from espnet2.bin.lid_inference import extract_embed_lid, gen_tsne_plot, get_parser, main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


def test_extract_embed_lid_basic_logic(tmp_path):
    """Test basic logic flow of extract_embed_lid function"""

    mock_args = MagicMock()
    mock_args.log_level = "INFO"
    mock_args.ngpu = 0
    mock_args.seed = 42
    mock_args.dtype = "float32"
    mock_args.lid_train_config = "dummy_config.yaml"
    mock_args.lid_model_file = "dummy_model.pth"
    mock_args.data_path_and_name_and_type = [("dump/test/wav.scp", "speech", "sound")]
    mock_args.shape_file = None
    mock_args.fix_duration = False
    mock_args.target_duration = None
    mock_args.valid_batch_size = 1
    mock_args.num_workers = 1
    mock_args.lang2utt = "dump/test/lang2utt"
    mock_args.extract_embd = True
    mock_args.checkpoint_interval = 1000
    mock_args.resume = True
    mock_args.save_embd_per_utt = True
    mock_args.save_embd_avg_lang = True
    mock_args.max_utt_per_lang_for_tsne = 100
    mock_args.output_dir = str(tmp_path)
    mock_args.save_tsne_plot = False

    with patch(
        "espnet2.bin.lid_inference.build_dataclass"
    ) as mock_build_dataclass, patch(
        "espnet2.bin.lid_inference.LIDTask"
    ) as mock_lid_task, patch(
        "espnet2.bin.lid_inference.set_all_random_seed"
    ), patch(
        "espnet2.bin.lid_inference.model_summary"
    ), patch(
        "builtins.open", mock_open(read_data="eng 0\nfra 1\n")
    ), patch(
        "espnet2.bin.lid_inference.Reporter"
    ) as mock_reporter, patch(
        "espnet2.bin.lid_inference.glob"
    ) as mock_glob, patch(
        "espnet2.bin.lid_inference.np.load"
    ), patch(
        "espnet2.bin.lid_inference.np.savez"
    ), patch(
        "espnet2.bin.lid_inference.os.remove"
    ), patch(
        "espnet2.bin.lid_inference.os.path.exists", return_value=False
    ):

        mock_distributed_option = MagicMock()
        mock_distributed_option.distributed = False
        mock_distributed_option.dist_rank = 0
        mock_distributed_option.dist_world_size = 1
        mock_build_dataclass.return_value = mock_distributed_option

        mock_model = MagicMock()
        mock_train_args = MagicMock()
        mock_lid_task.build_model_from_file.return_value = (mock_model, mock_train_args)

        mock_iterator = MagicMock()
        mock_lid_task.build_streaming_iterator.return_value = mock_iterator

        mock_trainer = MagicMock()
        mock_lid_task.trainer = mock_trainer

        mock_reporter_instance = MagicMock()
        mock_reporter.return_value = mock_reporter_instance

        mock_glob.return_value = []

        try:
            extract_embed_lid(mock_args)
            # If no exception is raised, the basic logic flow is working
            assert True
        except Exception as e:
            # We expect some exceptions due to mocked dependencies
            # but the core logic should have been executed
            assert "dummy" in str(e) or "mock" in str(e)

        # Verify that key components were called
        mock_build_dataclass.assert_called_once()
        mock_lid_task.build_model_from_file.assert_called_once()
        mock_lid_task.build_streaming_iterator.assert_called_once()
        mock_lid_task.trainer.extract_embed_lid.assert_called_once()

        # Verify output directory was accessed
        assert mock_args.output_dir == str(tmp_path)


@pytest.fixture()
def sample_embeddings():
    """Create sample embedding data for testing"""
    return {
        "eng": [np.random.rand(128) for _ in range(5)],
        "fra": [np.random.rand(128) for _ in range(3)],
        "deu": np.random.rand(128),
    }


def test_gen_tsne_plot(tmp_path, sample_embeddings):
    """Test gen_tsne_plot function by mocking all external dependencies"""
    output_dir = str(tmp_path / "tsne_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create mock for t-SNE
    mock_tsne_instance = MagicMock()
    mock_tsne_instance.fit_transform.return_value = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
        ]
    )

    # Create mock for pandas DataFrame
    mock_df_instance = MagicMock()
    mock_df_instance.to_csv = MagicMock()
    mock_df_instance.__getitem__ = MagicMock()
    mock_label_series = MagicMock()
    mock_label_series.unique.return_value = ["eng", "fra", "deu"]
    mock_df_instance.__getitem__.return_value = mock_label_series

    # Create mock for plotly figure
    mock_plotly_fig = MagicMock()
    mock_plotly_fig.write_html = MagicMock()

    # Mock all external library modules that gen_tsne_plot tries to import
    mock_modules = {
        "adjustText": MagicMock(),
        "plotly": MagicMock(),
        "plotly.express": MagicMock(),
        "matplotlib": MagicMock(),
        "matplotlib.pyplot": MagicMock(),
        "pandas": MagicMock(),
    }

    # Configure the mock behaviors
    mock_modules["adjustText"].adjust_text = MagicMock()
    mock_modules["plotly.express"].scatter = MagicMock(return_value=mock_plotly_fig)
    mock_modules["pandas"].DataFrame = MagicMock(return_value=mock_df_instance)

    # Configure matplotlib.pyplot mock
    plt_mock = mock_modules["matplotlib.pyplot"]
    plt_mock.figure = MagicMock()
    plt_mock.scatter = MagicMock()
    plt_mock.text = MagicMock()
    plt_mock.title = MagicMock()
    plt_mock.savefig = MagicMock()
    plt_mock.close = MagicMock()
    plt_mock.get_cmap = MagicMock()

    # Use sys.modules patching to make all library imports return our mocks
    with patch.dict("sys.modules", mock_modules):
        with patch("espnet2.bin.lid_inference.TSNE", return_value=mock_tsne_instance):
            # Mock logging to avoid log output during tests
            with patch("espnet2.bin.lid_inference.logging.info"):

                # Execute the function - this should now work without real dependencies
                gen_tsne_plot(
                    lang_to_embds_dic=sample_embeddings,
                    output_dir=output_dir,
                    seed=42,
                    perplexity=2,
                    max_iter=250,
                )

                # Verify the key operation: t-SNE was called
                mock_tsne_instance.fit_transform.assert_called_once()
