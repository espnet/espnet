"""CPU tests for speechlm/parallel_utils/pipeline.py."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torchtitan", reason="torchtitan not installed")


class TestBuildPipeline:
    def test_single_stage_path(self):
        """Exercise the single-stage 1F1B schedule path.

        Uses patched PipelineStage and schedule class so no distributed
        initialization is required.
        """
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        # Stage model with the attributes build_pipeline reads.
        stage_model = nn.Linear(4, 4)
        stage_model.pp_rank = 0
        stage_model.pp_degree = 2
        stage_model.is_last_stage = True

        # Fake ParallelDims + pp mesh.
        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        # Fake schedule class: single-stage.
        class _FakeSchedule:
            def __init__(self, stage, n_microbatches, loss_fn, scale_grads):
                self.stage = stage
                self.n_microbatches = n_microbatches
                self.loss_fn = loss_fn
                self.scale_grads = scale_grads

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeSchedule),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
            patch.object(
                pipeline,
                "PipelineScheduleMulti",
                new=type("_DummyMulti", (), {}),
            ),
        ):
            schedule, has_last = pipeline.build_pipeline(
                stage_model,
                parallel_dims=pd,
                titan_config={"pp_schedule": "1F1B"},
                n_microbatches=4,
            )
        assert isinstance(schedule, _FakeSchedule)
        assert has_last is True
        # Verify _identity_loss pass-through behavior
        t = torch.tensor(3.14)
        assert schedule.loss_fn(t, None) is t
        assert schedule.loss_fn((t,), None) is t

    def test_single_stage_unwraps_list_of_one(self):
        """An nn.ModuleList with one chunk is unwrapped for single-stage.

        A single-stage schedule given a one-chunk ModuleList proceeds as if
        the chunk were passed directly.
        """
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        stage = nn.Linear(4, 4)
        stage.pp_rank = 0
        stage.pp_degree = 1
        stage.is_last_stage = True
        model_list = nn.ModuleList([stage])

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        class _FakeSchedule:
            def __init__(self, stage, **kw):
                self.stage = stage

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeSchedule),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
            patch.object(
                pipeline,
                "PipelineScheduleMulti",
                new=type("_DummyMulti", (), {}),
            ),
        ):
            schedule, _ = pipeline.build_pipeline(
                model_list,
                parallel_dims=pd,
                titan_config={"pp_schedule": "1F1B"},
                n_microbatches=2,
            )
        assert schedule is not None

    def test_multi_stage_path(self):
        """Exercise the multi-stage Interleaved1F1B path."""
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        # Two virtual chunks on this rank.
        chunk0 = nn.Linear(4, 4)
        chunk0.stage_idx = 0
        chunk0.num_virtual_stages = 2
        chunk0.is_last_stage = False
        chunk1 = nn.Linear(4, 4)
        chunk1.stage_idx = 1
        chunk1.num_virtual_stages = 2
        chunk1.is_last_stage = True
        chunks = nn.ModuleList([chunk0, chunk1])

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        # Multi-stage schedule: subclass of PipelineScheduleMulti.
        from torch.distributed.pipelining.schedules import PipelineScheduleMulti

        class _FakeMulti(PipelineScheduleMulti):
            def __init__(self, stages, n_microbatches, loss_fn, scale_grads):
                self.stages = stages
                self.n_microbatches = n_microbatches
                self.loss_fn = loss_fn

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeMulti),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
        ):
            schedule, has_last = pipeline.build_pipeline(
                chunks,
                parallel_dims=pd,
                titan_config={"pp_schedule": "Interleaved1F1B"},
                n_microbatches=4,
            )
        assert has_last is True  # chunk1.is_last_stage
        assert schedule.n_microbatches == 4
        # Identity loss handles both tuple and tensor outputs.
        t = torch.tensor(1.0)
        assert schedule.loss_fn(t, None) is t

    def test_multi_stage_requires_divisible_microbatches(self):
        """n_microbatches must be divisible by vpp_degree."""
        from unittest.mock import MagicMock, patch

        from torch.distributed.pipelining.schedules import PipelineScheduleMulti

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        chunks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
        for i, c in enumerate(chunks):
            c.stage_idx = i
            c.num_virtual_stages = 3
            c.is_last_stage = i == 2

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        class _FakeMulti(PipelineScheduleMulti):
            def __init__(self, *a, **k):
                pass

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeMulti),
            patch.object(pipeline, "PipelineStage", return_value=MagicMock()),
        ):
            # 3 chunks, 4 microbatches → not divisible.
            with pytest.raises(AssertionError, match="divisible"):
                pipeline.build_pipeline(
                    chunks,
                    parallel_dims=pd,
                    titan_config={"pp_schedule": "Interleaved1F1B"},
                    n_microbatches=4,
                )

    def test_single_stage_rejects_multi_chunk_list(self):
        """Multi-chunk list with a single-stage schedule should fail."""
        from unittest.mock import MagicMock, patch

        from espnet2.speechlm.model.speechlm.parallel_utils import pipeline

        chunks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

        pd = MagicMock()
        mesh = MagicMock()
        mesh.get_group.return_value = None
        pd.get_mesh.return_value = mesh

        class _FakeSchedule:
            def __init__(self, *a, **k):
                pass

        with (
            patch.object(pipeline, "get_schedule_class", return_value=_FakeSchedule),
            patch.object(
                pipeline,
                "PipelineScheduleMulti",
                new=type("_DummyMulti", (), {}),
            ),
        ):
            with pytest.raises(AssertionError, match="expects 1 model"):
                pipeline.build_pipeline(
                    chunks,
                    parallel_dims=pd,
                    titan_config={},
                    n_microbatches=2,
                )
