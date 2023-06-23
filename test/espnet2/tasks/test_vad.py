import tempfile
from argparse import Namespace

import pytest

from espnet2.bin.vad_scoring import scoring
from espnet2.tasks.vad import VADTask


def test_add_arguments():
    VADTask.get_parser()


def test_add_arguments_help():
    parser = VADTask.get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_main_help():
    with pytest.raises(SystemExit):
        VADTask.main(cmd=["--help"])


def test_main_print_config():
    with pytest.raises(SystemExit):
        VADTask.main(cmd=["--print_config"])


def test_main_with_no_args():
    with pytest.raises(SystemExit):
        VADTask.main(cmd=[])


def test_print_config_and_load_it(tmp_path):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        VADTask.print_config(f)
    parser = VADTask.get_parser()
    parser.parse_args(["--config", str(config_file)])


def get_dummy_namespace():
    return Namespace(
        pre_postencoder_norm=False,
        input_size=40,
        frontend="frontend",
        frontend_conf={"n_fft": 51, "win_length": 40, "hop_length": 16},
        specaug="specaug",
        specaug_conf={"apply_time_warp": True, "time_mask_width_range": 4},
        normalize=None,
        normalize_conf=None,
        encoder="rnn",
        encoder_conf={
            "rnn_type": "gru",
            "bidirectional": True,
            "use_projection": True,
            "num_layers": 4,
            "hidden_size": 320,
            "output_size": 320,
            "dropout": 0.2,
        },
        init=None,
        model_conf={},
        segment_length=10.0,
    )


def test_build_model():
    args = get_dummy_namespace()

    _ = VADTask.build_model(args)


def test_build_collate_fn():
    args = get_dummy_namespace()

    _ = VADTask.build_collate_fn(args, True)


@pytest.mark.parametrize("use_preprocessor", [True, False])
def test_build_preprocess_fn(use_preprocessor):
    args = get_dummy_namespace()

    new_args = {
        "use_preprocessor": use_preprocessor,
    }
    args.__dict__.update(new_args)

    _ = VADTask.build_preprocess_fn(args, True)


@pytest.mark.parametrize("inference", [True, False])
def test_required_data_names(inference):
    retval = VADTask.required_data_names(True, inference)

    if inference:
        assert retval == ("speech",)
    else:
        assert retval == ("speech", "text")


@pytest.mark.parametrize("inference", [True, False])
def test_optional_data_names(inference):
    retval = VADTask.optional_data_names(True, inference)

    assert retval == ("transcript",)


def test_scoring(capfd):
    # Case 1
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as hyp_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as ref_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as output_file:
        hyp_file.writelines(
            [
                "AMI_EN2002a_H00_0000 0.0 0.14 0.97 5.36 6.7 6.91\n",
                "AMI_EN2002a_H00_0020 0.0 0.14 1.48 3.64 5.54 10.0\n",
                "AMI_EN2002a_H00_0030 0.0 0.14 2.62 2.84\n",
                "AMI_EN2002a_H00_0100 0.0 0.14\n",
                "AMI_EN2002a_H00_0180 0.0 0.14 2.74 4.77 5.51 6.12\n",
            ]
        )
        ref_file.writelines(
            [
                "AMI_EN2002a_H00_0000 0.96 6.85\n",
                "AMI_EN2002a_H00_0020 1.47 3.29 5.5 10\n",
                "AMI_EN2002a_H00_0030 0 0.16 2.58 2.71\n",
                "AMI_EN2002a_H00_0100 6.69 6.87\n",
                "AMI_EN2002a_H00_0180 1.89 5.98\n",
            ]
        )

        # Ensure the data is written to the file
        hyp_file.flush()
        ref_file.flush()

        # Call the scoring function
        scoring(hyp_file.name, ref_file.name, output_file.name)

        # Check the output file
        output_file.seek(0)  # reset the file pointer to the start of the file
        output_data = output_file.read()

        expected_output = "|Precision: 0.916|\n|Recall: 0.8068|\n|F1_score: 0.858|\n"
        assert output_data == expected_output, "Output data is not as expected."

    # Case 2
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as hyp_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as ref_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=True
    ) as output_file:
        hyp_file.writelines(
            [
                "AMI_TS3003d_H03_2430 0.02 1.26 5.32 5.68\n",
                "AMI_TS3003d_H03_2440 0.0 0.14\n",
                "AMI_TS3003d_H03_2460 0.0 0.14 1.81 2.17\n",
                "AMI_TS3003d_H03_2480 0.0 0.14\n",
                "AMI_TS3003d_H03_2530 0.0 0.14 0.67 0.84\n",
                "AMI_TS3003d_H03_2550 0.0 0.14 1.55 2.96\n",
                "AMI_TS3003d_H03_2560 0.0 0.14 4.26 6.95\n",
                "AMI_TS3003d_H03_2570 0.0 0.14\n",
                "AMI_TS3003d_H03_2580 0.0 0.14 6.93 7.57\n",
                "AMI_TS3003d_H03_2590 0.02 0.19 0.62 1.54",
            ]
        )
        ref_file.writelines(
            [
                "AMI_TS3003d_H03_2430 0 1.21 5.3 5.57\n",
                "AMI_TS3003d_H03_2440 4.79 5.21\n",
                "AMI_TS3003d_H03_2460 1.76 2.09\n",
                "AMI_TS3003d_H03_2480 2.15 2.63\n",
                "AMI_TS3003d_H03_2530 0.17 1.06\n",
                "AMI_TS3003d_H03_2550 1.48 5.15\n",
                "AMI_TS3003d_H03_2560 4.21 6.88\n",
                "AMI_TS3003d_H03_2570 0.62 1.02\n",
                "AMI_TS3003d_H03_2580 6.6 7.42\n",
                "AMI_TS3003d_H03_2590 0.69 1.56",
            ]
        )

        # Ensure the data is written to the file
        hyp_file.flush()
        ref_file.flush()

        # Call the scoring function
        scoring(hyp_file.name, ref_file.name, output_file.name)

        # Check the output file
        output_file.seek(0)  # reset the file pointer to the start of the file
        output_data = output_file.read()

        expected_output = "|Precision: 0.7996|\n|Recall: 0.6035|\n|F1_score: 0.6878|\n"
        assert output_data == expected_output, "Output data is not as expected."
