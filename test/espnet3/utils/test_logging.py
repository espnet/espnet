from __future__ import annotations

import io
import logging as py_logging
from pathlib import Path

import torch
import torch.nn as nn

from espnet3.utils import logging_utils as elog

# | Test Name                                              | Description                                                    | # noqa: E501
# |-------------------------------------------------------|----------------------------------------------------------------| # noqa: E501
# | test_configure_logging_adds_console_and_file          | Adds both console and file handlers and writes log file        | # noqa: E501
# | test_configure_logging_is_idempotent                  | Repeated configure keeps a single file handler                 | # noqa: E501
# | test_configure_logging_rotates_existing_file          | Rotates pre-existing log file before new logging               | # noqa: E501
# | test_set_log_format_updates_globals_and_handlers      | Updates global log/date formats and handler formatter          | # noqa: E501
# | test_log_run_metadata_logs_command_and_git            | Logs argv, config paths, and git metadata                      | # noqa: E501
# | test_log_run_metadata_writes_requirements             | Writes pip freeze output to requirements.txt                   | # noqa: E501
# | test_log_training_summary_includes_model_and_optimizer| Logs model, optimizer, and scheduler summaries                 | # noqa: E501
# | test_log_data_organizer_includes_datasets             | Logs train/valid dataset summaries from DataOrganizer          | # noqa: E501
# | test_log_data_organizer_combined_variants             | Logs CombinedDataset variants with custom transforms           | # noqa: E501
# | test_log_dataloader_formats_human_readable            | Formats DataLoader info with key attributes                    | # noqa: E501
# | test_log_dataloader_iter_factory_includes_batch_sampler_repr | Logs iterator factory with batch sampler repr           | # noqa: E501
# | test_build_qualified_name_for_objects_and_classes     | Builds qualified names for objects, classes, and collections   | # noqa: E501
# | test_build_callable_name_for_functions_and_callables  | Builds callable names for functions and callable classes       | # noqa: E501


def _reset_logger(logger: py_logging.Logger, handlers, level, propagate: bool) -> None:
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_configure_logging_adds_console_and_file(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        logger = elog.configure_logging(log_dir=tmp_path, filename="run.log")
        logger.info("hello")

        file_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.FileHandler)
        ]
        stream_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.StreamHandler)
        ]

        assert (tmp_path / "run.log").exists()
        assert len(file_handlers) == 1
        assert len(stream_handlers) >= 1
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_configure_logging_is_idempotent(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        elog.configure_logging(log_dir=tmp_path, filename="run.log")
        elog.configure_logging(log_dir=tmp_path, filename="run.log")

        file_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.FileHandler)
        ]
        assert len(file_handlers) == 1
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_configure_logging_rotates_existing_file(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        log_path = tmp_path / "run.log"
        log_path.write_text("old log\n", encoding="utf-8")

        elog.configure_logging(log_dir=tmp_path, filename="run.log")

        rotated = tmp_path / "run1.log"
        assert rotated.exists()
        assert rotated.read_text(encoding="utf-8") == "old log\n"
        assert log_path.exists()
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_set_log_format_updates_globals_and_handlers():
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    old_log_format = elog.LOG_FORMAT
    old_date_format = elog.DATE_FORMAT
    try:
        root.handlers = []
        root.propagate = False
        handler = py_logging.StreamHandler()
        root.addHandler(handler)

        new_fmt = "%(levelname)s %(message)s"
        new_date = "%Y"
        elog.set_log_format(log_format=new_fmt, date_format=new_date, apply=True)

        assert elog.LOG_FORMAT == new_fmt
        assert elog.DATE_FORMAT == new_date
        assert handler.formatter is not None
        assert handler.formatter._fmt == new_fmt
        assert handler.formatter.datefmt == new_date
    finally:
        elog.LOG_FORMAT = old_log_format
        elog.DATE_FORMAT = old_date_format
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_log_run_metadata_logs_command_and_git(monkeypatch, tmp_path: Path):
    logger = py_logging.getLogger("espnet3.test.logging")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    stream = io.StringIO()
    handler = py_logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False

    monkeypatch.setattr(
        elog, "get_git_metadata", lambda cwd=None: {"commit": "abc", "branch": "main"}
    )

    try:
        elog.log_run_metadata(
            logger,
            argv=["run.py", "--arg", "1"],
            configs={"train": tmp_path / "train.yaml"},
        )
        out = stream.getvalue()
        assert "run.py --arg 1" in out
        assert str(tmp_path) in out
        assert "train.yaml" in out
        assert "commit=abc" in out and "branch=main" in out
        assert "Python:" in out
    finally:
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_run_metadata_writes_requirements(monkeypatch, tmp_path: Path):
    logger = py_logging.getLogger("espnet3.test.requirements")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    handler = py_logging.FileHandler(tmp_path / "run.log")
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False

    monkeypatch.setattr(elog, "_run_pip_freeze", lambda: "pkg==1.2.3")

    try:
        elog.log_run_metadata(logger, write_requirements=True)
        contents = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
        assert "pkg==1.2.3" in contents
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {"x": torch.tensor([idx], dtype=torch.float32)}

    def __len__(self):
        return 3


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def _capture_logger(name: str):
    logger = py_logging.getLogger(name)
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    stream = io.StringIO()
    handler = py_logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False
    return logger, stream, (old_handlers, old_level, old_propagate, handler)


def test_log_training_summary_includes_model_and_optimizer():
    logger, stream, cleanup = _capture_logger("espnet3.test.train_summary")
    old_handlers, old_level, old_propagate, handler = cleanup

    from espnet3.components.modeling.lightning_module import ESPnetLightningModule

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    module = ESPnetLightningModule.__new__(ESPnetLightningModule)
    try:
        module._log_training_summary(
            logger, model, optimizer=optimizer, scheduler=scheduler
        )
        out = stream.getvalue()
        assert "Model summary:" in out
        assert "Class Name: DummyModel" in out
        assert "Optimizer[0]:" in out
        assert "Scheduler[0]:" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_data_organizer_includes_datasets():
    logger, stream, cleanup = _capture_logger("espnet3.test.data_organizer")
    old_handlers, old_level, old_propagate, handler = cleanup

    from espnet3.components.data.data_organizer import DataOrganizer
    from espnet3.components.data.dataset import CombinedDataset

    class DummyOrganizer(DataOrganizer):
        def __init__(self):
            self.preprocessor = lambda x: x
            self.train = CombinedDataset([DummyDataset()], [(lambda x: x, lambda x: x)])
            self.valid = CombinedDataset([DummyDataset()], [(lambda x: x, lambda x: x)])
            self.test_sets = {}

    try:
        DummyOrganizer().log_summary(logger)
        out = stream.getvalue()
        assert "Data organizer:" in out
        assert "train dataset:" in out
        assert "valid dataset:" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_data_organizer_combined_variants():
    logger, stream, cleanup = _capture_logger("espnet3.test.data_organizer.variants")
    old_handlers, old_level, old_propagate, handler = cleanup

    def custom_transform(sample):
        return sample

    def other_transform(sample):
        return sample

    def custom_preprocessor(sample):
        return sample

    from espnet3.components.data.data_organizer import DataOrganizer
    from espnet3.components.data.dataset import CombinedDataset

    class DummyOrganizer(DataOrganizer):
        def __init__(self):
            self.preprocessor = custom_preprocessor
            self.train = CombinedDataset(
                [DummyDataset(), DummyDataset()],
                [(custom_transform, custom_preprocessor), (other_transform, None)],
            )
            self.valid = CombinedDataset(
                [DummyDataset()],
                [(custom_transform, None)],
            )
            self.test_sets = {}

    try:
        DummyOrganizer().log_summary(logger)
        out = stream.getvalue()
        assert "Data organizer:" in out
        assert "train dataset:" in out
        assert "valid dataset:" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_dataloader_formats_human_readable():
    logger, stream, cleanup = _capture_logger("espnet3.test.dataloader")
    old_handlers, old_level, old_propagate, handler = cleanup

    from espnet3.components.data import dataloader as espnet3_dataloader

    espnet3_dataloader._LOGGED_DATALOADER = set()
    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2, num_workers=0)
    try:
        espnet3_dataloader.log_dataloader(logger, loader, label="train")
        out = stream.getvalue()
        assert "DataLoader[train] class:" in out
        assert "batch_size" in out
        assert "num_workers" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_dataloader_iter_factory_includes_batch_sampler_repr():
    logger, stream, cleanup = _capture_logger("espnet3.test.dataloader.iter_factory")
    old_handlers, old_level, old_propagate, handler = cleanup

    from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
    from espnet2.samplers.build_batch_sampler import build_batch_sampler
    from espnet3.components.data import dataloader as espnet3_dataloader

    batches = build_batch_sampler(
        shape_files=["test_utils/espnet3/stats/stats_dummy"],
        type="unsorted",
        batch_size=2,
        batch_bins=4000000,
    )
    iter_factory = SequenceIterFactory(DummyDataset(), batches=batches, shuffle=False)
    iterator = iter_factory.build_iter(0, shuffle=False)

    try:
        espnet3_dataloader._LOGGED_DATALOADER = set()
        espnet3_dataloader.log_dataloader(logger, iterator, label="train")
        out = stream.getvalue()
        assert "DataLoader[train] class:" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_build_qualified_name_for_objects_and_classes():
    from espnet3.utils.logging_utils import build_qualified_name

    class LocalClass:
        pass

    assert build_qualified_name(LocalClass).endswith(".LocalClass")
    assert build_qualified_name(LocalClass()).endswith(".LocalClass")
    assert build_qualified_name([1, 2]).startswith("list(len=")


def test_build_callable_name_for_functions_and_callables():
    from espnet3.utils.logging_utils import build_callable_name

    def local_fn():
        return None

    class CallableClass:
        def __call__(self):
            return None

    assert build_callable_name(local_fn).endswith(".local_fn")
    assert build_callable_name(CallableClass).endswith(".CallableClass")
    assert build_callable_name(CallableClass()).endswith(".CallableClass")
