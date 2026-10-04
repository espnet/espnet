"""CPU tests for speechlm/parallel_utils/parallel_dims.py."""

import pytest

pytest.importorskip("torchtitan", reason="torchtitan not installed")


class TestInitParallelDims:
    """Tests for init_parallel_dims.

    The function calls ``parallel_dims.build_mesh()`` which requires
    ``torch.distributed`` initialized. We patch ``build_mesh`` to a
    no-op so the real ParallelDims constructor still runs but the
    mesh-build step is skipped, keeping these tests dist-init free.
    """

    def test_parses_titan_config(self):
        """init_parallel_dims forwards titan_config keys to ParallelDims.

        Returns (parallel_dims, local_rank, global_rank).
        """
        from unittest.mock import patch

        from torchtitan.distributed import ParallelDims

        from espnet2.speechlm.model.speechlm.parallel_utils.parallel_dims import (
            init_parallel_dims,
        )

        with (
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.dist.get_rank",
                return_value=1,
            ),
            patch(
                "espnet2.speechlm.model.speechlm.parallel_utils."
                "parallel_dims.torch.cuda.current_device",
                return_value=2,
            ),
            patch.object(ParallelDims, "build_mesh", lambda self: None),
        ):
            pd, local_rank, global_rank = init_parallel_dims(
                {"dp_replicate": 1, "dp_shard": -1, "pp_degree": 1}
            )
        assert local_rank == 2
        assert global_rank == 1
        assert pd.world_size == 4
        # dp_shard=-1 auto-computes to world_size when other dims are 1
        assert pd.dp_shard == 4

    def test_rejects_expert_parallelism(self):
        from espnet2.speechlm.model.speechlm.parallel_utils.parallel_dims import (
            init_parallel_dims,
        )

        with pytest.raises(ValueError, match="does not support expert parallelism"):
            init_parallel_dims({"ep": 4})
