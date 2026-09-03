"""Tests for espnet2/speechlm/model/__init__.py — Model Registry.

``SpeechLMJobTemplate`` transitively imports ``lm/loss.py`` (via
``lm/parallel.py``), which pulls in ``liger_kernel`` at module load
time. Liger is not a declared project dep; the whole file is skipped
(via the sibling ``conftest.py``'s ``collect_ignore_glob``) when
``liger_kernel.ops.fused_linear_cross_entropy`` is not importable.
"""

from espnet2.speechlm.model import _all_job_types
from espnet2.speechlm.model.abs_job import AbsJobTemplate
from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate


def test_all_job_types_exists():
    assert isinstance(_all_job_types, dict)
    assert len(_all_job_types) > 0


def test_speechlm_in_registry():
    assert "speechlm" in _all_job_types
    assert _all_job_types["speechlm"] is SpeechLMJobTemplate


def test_all_values_are_subclass():
    for name, cls in _all_job_types.items():
        assert issubclass(
            cls, AbsJobTemplate
        ), f"Registry entry '{name}' is not a subclass of AbsJobTemplate"
