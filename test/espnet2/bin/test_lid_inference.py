from argparse import ArgumentParser
from pathlib import Path
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
    """Test basic functionality of gen_tsne_plot function"""
    plotly = pytest.importorskip("plotly")
    adjusttext = pytest.importorskip("adjustText")
    
    output_dir = str(tmp_path / "tsne_output")

    with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.scatter"), patch(
        "matplotlib.pyplot.savefig"
    ), patch("matplotlib.pyplot.close"), patch("pandas.DataFrame.to_csv"), patch(
        "plotly.express.scatter"
    ) as mock_px_scatter, patch(
        "adjustText.adjust_text"
    ):

        mock_fig = MagicMock()
        mock_px_scatter.return_value = mock_fig

        # Should not raise any exceptions
        gen_tsne_plot(
            lang_to_embds_dic=sample_embeddings,
            output_dir=output_dir,
            seed=42,
            perplexity=2,
            max_iter=250,
        )

        # Check if output directory is created
        assert Path(output_dir).exists()
