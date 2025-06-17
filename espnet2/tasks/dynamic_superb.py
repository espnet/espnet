import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch

from espnet2.tasks.abs_task import AbsTask
from espnet2.dynamic_superb.espnet_model import ESPnetQwen2AudioModel
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.train.preprocessor import Qwen2AudioPreprocessor

class DynamicSuperbTask(AbsTask):
    """Task class for Qwen2-Audio integration following ESPnet2 architecture[8]"""
    
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
        """Add task-specific arguments[8]"""
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
        """Build collate function[8]"""
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=-1,
        )
    
    @classmethod
    def build_preprocess_fn(cls, args: argparse.Namespace, train: bool) -> Optional[Callable]:
        """Build preprocessing function[8]"""
        return Qwen2AudioPreprocessor()
    
    @classmethod
    def required_data_names(cls, train: bool = True, inference: bool = False) -> Tuple[str, ...]:
        """Define required data names[8]"""
        if inference:
            return ("speech",)
        else:
            return ("speech", "text")
    
    @classmethod
    def optional_data_names(cls, train: bool = True, inference: bool = False) -> Tuple[str, ...]:
        """Define optional data names[8]"""
        return ()
    
    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetQwen2AudioModel:
        """Build the Qwen2-Audio model[8]"""
        model = ESPnetQwen2AudioModel(
            model_name=args.model_name,
        )
        return model
