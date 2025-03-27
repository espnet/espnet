# ESPnet-EZ Task class
# This class is a wrapper for Task classes to support custom datasets.
import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.category_iter_factory import CategoryIterFactory
from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.iterators.multiple_iter_factory import MultipleIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.samplers.build_batch_sampler import build_batch_sampler
from espnet2.samplers.category_balanced_sampler import CategoryBalancedSampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import AbsTask, IteratorOptions
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.asr_transducer import ASRTransducerTask
from espnet2.tasks.asvspoof import ASVSpoofTask
from espnet2.tasks.diar import DiarizationTask
from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
from espnet2.tasks.gan_svs import GANSVSTask
from espnet2.tasks.gan_tts import GANTTSTask
from espnet2.tasks.hubert import HubertTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.mt import MTTask
from espnet2.tasks.s2st import S2STTask
from espnet2.tasks.s2t import S2TTask
from espnet2.tasks.slu import SLUTask
from espnet2.tasks.spk import SpeakerTask
from espnet2.tasks.st import STTask
from espnet2.tasks.svs import SVSTask
from espnet2.tasks.tts import TTSTask
from espnet2.tasks.uasr import UASRTask
from espnet2.train.distributed_utils import DistributedOption

TASK_CLASSES = dict(
    asr=ASRTask,
    asr_transducer=ASRTransducerTask,
    asvspoof=ASVSpoofTask,
    diar=DiarizationTask,
    enh_s2t=EnhS2TTask,
    enh_tse=TargetSpeakerExtractionTask,
    enh=EnhancementTask,
    gan_svs=GANSVSTask,
    gan_tts=GANTTSTask,
    hubert=HubertTask,
    lm=LMTask,
    mt=MTTask,
    s2st=S2STTask,
    s2t=S2TTask,
    slu=SLUTask,
    spk=SpeakerTask,
    st=STTask,
    svs=SVSTask,
    tts=TTSTask,
    uasr=UASRTask,
)


def get_ez_task(task_name: str, use_custom_dataset: bool = False) -> AbsTask:
    """
    Retrieve a customized task class for the ESPnet-EZ framework.

    This function returns a task class based on the specified task name.
    If the `use_custom_dataset` flag is set to True, a version of the task
    class that supports custom datasets will be returned. The returned class
    inherits from the appropriate base task class and may be extended with
    additional functionality.

    Args:
        task_name (str): The name of the task to retrieve. This must be one of
            the keys defined in the `TASK_CLASSES` dictionary, such as 'asr',
            'mt', 'tts', etc.
        use_custom_dataset (bool, optional): A flag indicating whether to use
            a version of the task class that supports custom datasets. Defaults
            to False.

    Returns:
        AbsTask: An instance of the task class corresponding to the provided
        `task_name`. If `use_custom_dataset` is True, the returned class will
        be capable of handling custom datasets.

    Raises:
        KeyError: If `task_name` is not found in the `TASK_CLASSES` dictionary.

    Examples:
        >>> asr_task = get_ez_task("asr")
        >>> custom_asr_task = get_ez_task("asr", use_custom_dataset=True)

        >>> mt_task = get_ez_task("mt")
        >>> custom_mt_task = get_ez_task("mt", use_custom_dataset=True)

    Note:
        The task classes are designed to be used within the ESPnet-EZ framework,
        which allows for flexibility in handling various speech and language tasks.
        Ensure that the required dependencies for the specific task are properly
        installed and configured.
    """
    task_class = TASK_CLASSES[task_name]

    if use_custom_dataset:
        return get_ez_task_with_dataset(task_name)

    class ESPnetEZTask(task_class):
        build_model_fn = None

        @classmethod
        def build_model(cls, args=None):
            if cls.build_model_fn is not None:
                return cls.build_model_fn(args=args)
            else:
                return task_class.build_model(args=args)

    return ESPnetEZTask


def get_ez_task_with_dataset(task_name: str) -> AbsTask:
    """
    Create an ESPnet-EZ task class with a custom dataset for a given task.

    This function returns a task class that inherits from the specified
    task class in the ESPnet framework, enabling the use of custom datasets
    for training and validation. The created task class includes methods
    for building models and iterators specifically tailored for handling
    datasets.

    Args:
        task_name (str): The name of the task for which the class is being created.
                          This should correspond to one of the predefined task classes
                          in the ESPnet framework, such as 'asr', 'tts', etc.

    Returns:
        AbsTask: A subclass of AbsTask that supports custom datasets for the
            specified task.

    Examples:
        >>> from espnetez.task import get_ez_task_with_dataset
        >>> custom_asr_task = get_ez_task_with_dataset("asr")
        >>> custom_asr_task.train_dataset = my_custom_train_dataset
        >>> custom_asr_task.valid_dataset = my_custom_valid_dataset
        >>> model = custom_asr_task.build_model(args)
        >>> iterator = custom_asr_task.build_iter_factory(args, distributed_option,
            mode='train')

    Note:
        Ensure that the specified task name is valid and that the corresponding
        task class is available in the TASK_CLASSES dictionary. The created
        task class will need to have its `train_dataset` and `valid_dataset`
        attributes set to the appropriate dataset instances before training.
    """
    task_class = TASK_CLASSES[task_name]

    class ESPnetEZDataTask(task_class):
        build_model_fn = None
        train_dataset = None
        valid_dataset = None
        train_dataloader = None
        valid_dataloader = None

        @classmethod
        def build_model(cls, args=None):
            if cls.build_model_fn is not None:
                return cls.build_model_fn(args=args)
            else:
                return task_class.build_model(args=args)

        @classmethod
        def build_preprocess_fn(cls, *args, **kwargs) -> IteratorOptions:
            """Build a preprocess function for the task.
            When developers uses the ESPnetEZDataTask, developers should perform
            preprocess steps inside the custom dataset class.
            """
            return None

        @classmethod
        def build_iter_factory(
            cls,
            args: argparse.Namespace,
            distributed_option: DistributedOption,
            mode: str,
            kwargs: dict = None,
        ) -> AbsIterFactory:
            if mode == "train" and cls.train_dataloader is not None:
                return cls.train_dataloader
            elif mode == "valid" and cls.valid_dataloader is not None:
                return cls.valid_dataloader

            if mode == "valid" and args.valid_iterator_type is not None:
                iterator_type = args.valid_iterator_type
            else:
                iterator_type = args.iterator_type

            iter_options = cls.build_iter_options(
                args=args, distributed_option=distributed_option, mode=mode
            )
            # Overwrite iter_options if any kwargs is given
            if kwargs is not None:
                for k, v in kwargs.items():
                    setattr(iter_options, k, v)

            if cls.train_dataset is not None and cls.valid_dataset is not None:
                if iterator_type == "sequence":
                    return cls.build_sequence_iter_factory(
                        args=args,
                        iter_options=iter_options,
                        mode=mode,
                    )
                elif iterator_type == "category":
                    return cls.build_category_iter_factory(
                        args=args,
                        iter_options=iter_options,
                        mode=mode,
                    )
                elif iterator_type == "chunk":
                    return cls.build_chunk_iter_factory(
                        args=args,
                        iter_options=iter_options,
                        mode=mode,
                    )
                elif iterator_type == "task":
                    return cls.build_task_iter_factory(
                        args=args,
                        iter_options=iter_options,
                        mode=mode,
                    )
            else:
                return task_class.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode=mode,
                    kwargs=kwargs,
                )

        @classmethod
        @typechecked
        def build_sequence_iter_factory(
            cls,
            args: argparse.Namespace,
            iter_options: IteratorOptions,
            mode: str,
        ) -> AbsIterFactory:

            if mode == "train":
                dataset = cls.train_dataset
            elif mode == "valid" or mode == "plot_att":
                dataset = cls.valid_dataset
            else:
                raise ValueError(f"Invalid mode: {mode}")

            cls.check_task_requirements(
                dataset, args.allow_variable_data_keys, train=iter_options.train
            )

            # If you want to use utt2category_file, please use dump file.
            utt2category_file = None

            batch_sampler = build_batch_sampler(
                type=iter_options.batch_type,
                shape_files=iter_options.shape_files,
                fold_lengths=args.fold_length,
                batch_size=iter_options.batch_size,
                batch_bins=iter_options.batch_bins,
                sort_in_batch=args.sort_in_batch,
                sort_batch=args.sort_batch,
                drop_last=args.drop_last_iter,
                min_batch_size=(
                    torch.distributed.get_world_size()
                    if iter_options.distributed
                    else 1
                ),
                utt2category_file=utt2category_file,
            )

            batches = list(batch_sampler)
            if iter_options.num_batches is not None:
                batches = batches[: iter_options.num_batches]

            bs_list = [len(batch) for batch in batches]

            logging.info(f"[{mode}] dataset:\n{dataset}")
            logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
            logging.info(
                f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
                f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, "
                f"max={np.max(bs_list)}"
            )

            if iter_options.distributed:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                for batch in batches:
                    if len(batch) < world_size:
                        raise RuntimeError(
                            f"The batch-size must be equal or more than "
                            f"world_size: {len(batch)} < {world_size}"
                        )
                batches = [batch[rank::world_size] for batch in batches]

            return SequenceIterFactory(
                dataset=dataset,
                batches=batches,
                seed=args.seed,
                num_iters_per_epoch=iter_options.num_iters_per_epoch,
                shuffle=iter_options.train,
                shuffle_within_batch=args.shuffle_within_batch,
                num_workers=args.num_workers,
                collate_fn=iter_options.collate_fn,
                pin_memory=args.ngpu > 0,
            )

        @classmethod
        def build_category_iter_factory(
            cls,
            args: argparse.Namespace,
            iter_options: IteratorOptions,
            mode: str,
        ) -> AbsIterFactory:
            raise ValueError(
                "category2utt mandatory for category iterator, but not found."
                + "Please use dump file."
            )

        @classmethod
        def build_chunk_iter_factory(
            cls,
            args: argparse.Namespace,
            iter_options: IteratorOptions,
            mode: str,
        ) -> AbsIterFactory:
            raise NotImplementedError()

        @classmethod
        def build_task_iter_factory(
            cls,
            args: argparse.Namespace,
            iter_options: IteratorOptions,
            mode: str,
        ) -> AbsIterFactory:
            raise NotImplementedError

        @classmethod
        @typechecked
        def build_streaming_iterator(
            cls,
            data_path_and_name_and_type,
            preprocess_fn,
            collate_fn,
            key_file: Optional[str] = None,
            batch_size: int = 1,
            dtype: Optional[Any] = None,
            num_workers: int = 1,
            allow_variable_data_keys: bool = False,
            ngpu: int = 0,
            inference: bool = False,
            mode: Optional[str] = None,
            multi_task_dataset: bool = False,
        ) -> DataLoader:
            """Build DataLoader using iterable dataset.
            Basically this iterator is used in collect_stats stage.
            """
            if mode == "train" and cls.train_dataloader is not None:
                return cls.train_dataloader
            elif mode == "valid" and cls.valid_dataloader is not None:
                return cls.valid_dataloader

            # For backward compatibility for pytorch DataLoader
            if collate_fn is not None:
                kwargs = dict(collate_fn=collate_fn)
            else:
                kwargs = {}

            if mode == "train":
                ds = cls.train_dataset
            elif mode == "valid":
                ds = cls.valid_dataset
            else:
                raise ValueError(f"Invalid mode: {mode}")

            if hasattr(ds, "apply_utt2category") and ds.apply_utt2category:
                kwargs.update(batch_size=1)
            else:
                kwargs.update(batch_size=batch_size)

            cls.check_task_requirements(
                ds, allow_variable_data_keys, train=False, inference=inference
            )

            return DataLoader(
                dataset=ds,
                pin_memory=ngpu > 0,
                num_workers=num_workers,
                **kwargs,
            )

    return ESPnetEZDataTask
