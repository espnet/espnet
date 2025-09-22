import argparse
from typing import Callable, Optional, Tuple

from espnet2.ps2st.espnet_model import ESPnetQwen2AudioModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import Qwen2AudioPreprocessor
from espnet2.train.trainer import Trainer


class PS2STTask(AbsTask):
    """PS2ST refers to the prompt-based speech-to-speech/text task.

    The prompt is a text that serves as an instruction
    for the model to do a specific task such as ASR, IC, ST, etc.
    The output can be a text sequence or speech, depending on the task.
    For example, transcriptions for ASR, textual labels for classification,
    or synthesized speech for speech generation tasks.
    """

    num_optimizers: int = 1

    class_choices_list = [
        ClassChoices(
            name="model",
            classes=dict(
                qwen2_audio=ESPnetQwen2AudioModel,
            ),
            type_check=AbsESPnetModel,
            default="qwen2_audio",
        ),
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        """Add task-specific arguments"""
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--model_name",
            type=str,
            default="Qwen/Qwen2-Audio-7B-Instruct",
            help="Hugging Face model name for Qwen2-Audio",
        )

        # Add class choices
        for class_choices in cls.class_choices_list:
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable:
        """Build collate function"""
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=-1,
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable]:
        """Build preprocessing function"""
        return Qwen2AudioPreprocessor()

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define required data names"""
        if inference:
            return ("speech", "text")
        else:
            raise NotImplementedError

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define optional data names"""
        return ()

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetQwen2AudioModel:
        """Build the Qwen2-Audio model"""
        model = ESPnetQwen2AudioModel(
            model_name=args.model_name,
            decode_config_path=getattr(args, "decode_config_path", None),
        )
        return model
