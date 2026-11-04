import pytest
import torch

from espnet2.bin.tok_export import export_tokenizer_checkpoint


@pytest.mark.parametrize("prefix", ["tokenizer.", "module.tokenizer."])
def test_export_tokenizer_checkpoint(tmp_path, prefix):
    model_file = tmp_path / "joint.pth"
    output_file = tmp_path / "tokenizer.pth"
    torch.save(
        {
            "model": {
                f"{prefix}frontend.weight": torch.tensor([1.0]),
                f"{prefix}quantizer.centroids": torch.tensor([[2.0]]),
                "losses.asr.weight": torch.tensor([3.0]),
                "losses.reconstruction.generator.weight": torch.tensor([4.0]),
            }
        },
        model_file,
    )

    export_tokenizer_checkpoint(str(model_file), str(output_file))

    exported = torch.load(output_file, weights_only=True)
    assert set(exported["model"]) == {
        "tokenizer.frontend.weight",
        "tokenizer.quantizer.centroids",
    }


def test_export_tokenizer_checkpoint_rejects_missing_tokenizer(tmp_path):
    model_file = tmp_path / "joint.pth"
    torch.save({"losses.asr.weight": torch.tensor([1.0])}, model_file)

    with pytest.raises(RuntimeError, match=r"No tokenizer\.\* parameters"):
        export_tokenizer_checkpoint(str(model_file), str(tmp_path / "output.pth"))
