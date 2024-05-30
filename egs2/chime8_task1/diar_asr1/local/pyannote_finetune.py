import argparse
import os.path
import shutil
from pathlib import Path
from types import MethodType

from pyannote.audio import Inference, Model
from pyannote.audio.tasks import Segmentation
from pyannote.database import FileFinder, get_protocol
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from torch.optim import Adam
from torch_audiomentations import (
    AddColoredNoise,
    Compose,
    Gain,
    PeakNormalization,
    PolarityInversion,
)


def finetune_segmentation(
    n_gpus=1,
    max_epochs=40,
    batch_size=64,
    learning_rate=1e-5,
    gradient_clip=0.5,
    patience=10,
    num_workers=8,
    auth_token=None,
    exp_folder="./exp/pyannote_diarization_finetuned",
    ft_protocol="chime8_finetune.SpeakerDiarization.only_words",
):
    model = Model.from_pretrained("pyannote/segmentation", use_auth_token=auth_token)

    dataset = get_protocol(ft_protocol, {"audio": FileFinder()})

    augmentation = Compose(
        transforms=[  # using pitch-shifting and bandstopfiltering
            # slows significantly training, but you can try
            # as it will likely improve results a bit.
            Gain(
                min_gain_in_db=-6.0,
                max_gain_in_db=6.0,
                p=0.5,
            ),
            PeakNormalization(apply_to="only_too_loud_sounds"),
            PolarityInversion(p=0.5),
            AddColoredNoise(p=0.2),
        ],
        output_type="dict",
    )
    task = Segmentation(
        dataset,
        duration=model.specifications.duration,
        max_num_speakers=len(model.specifications.classes),
        batch_size=batch_size,
        num_workers=num_workers,
        loss="bce",
        vad_loss="bce",
        augmentation=augmentation,
    )
    model.task = task
    model.setup(stage="fit")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=learning_rate)

    model.configure_optimizers = MethodType(configure_optimizers, model)
    # we monitor diarization error rate on the validation set
    # and use to keep the best checkpoint and stop early
    monitor, direction = task.val_monitor
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        filename="{epoch}",
        verbose=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=0.0,
        patience=patience,
        strict=True,
        verbose=False,
    )

    callbacks = [RichProgressBar(), checkpoint, early_stopping]
    Path(exp_folder).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        accelerator="gpu",
        devices=list(range(n_gpus)),
        callbacks=callbacks,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip,
        default_root_dir=exp_folder,
    )
    trainer.fit(model)
    shutil.copyfile(
        checkpoint.best_model_path,
        os.path.join(Path(checkpoint.best_model_path).parent, "best.ckpt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine-tuning script for Pyannote segmentation model.",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Access token for HuggingFace Pyannote model."
        "see https://github.com/pyannote/pyannote-audio"
        "/blob/develop/tutorials/applying_a_pipeline.ipynb",
        metavar="STR",
        dest="auth_token",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use in fine-tuning.",
        metavar="INT",
        dest="ngpus",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Batch size to use in fine-tuning.",
        metavar="INT",
        dest="batch_size",
    )
    parser.add_argument(
        "--learning_rate",
        type=str,
        required=False,
        default="1e-5",
        help="Learning rate to use in fine-tuning.",
        metavar="STR",
        dest="learning_rate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=8,
        help="Num workers for dataloading.",
        metavar="INT",
        dest="num_workers",
    )
    parser.add_argument(
        "--exp_folder",
        type=str,
        required=False,
        default="./exp/pyannote_diarization_finetuned",
        help="Folder where to save the results and " "logs for the fine-tuning.",
        metavar="STR",
        dest="exp_folder",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=False,
        default="chime8_finetune.SpeakerDiarization.only_words",
        help="Dataset to use in fine-tuning, it must match an "
        "entry in the database.yml in this folder.",
        metavar="STR",
        dest="protocol",
    )

    args = parser.parse_args()
    finetune_segmentation(
        batch_size=args.batch_size,
        n_gpus=args.ngpus,
        num_workers=args.num_workers,
        auth_token=args.auth_token,
        learning_rate=float(args.learning_rate),
        exp_folder=args.exp_folder,
        ft_protocol=args.protocol,
    )
