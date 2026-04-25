# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Model module for SpeechLM job templates and configurations."""

from espnet2.speechlm.model.speechlm.speechlm_job import (
    SpeechLMJobTemplate,
    SpeechLMJobTemplateWithPrefix,
)

_all_job_types = {
    "speechlm": SpeechLMJobTemplate,
    "speechlm_with_prefix": SpeechLMJobTemplateWithPrefix,
}

__all__ = [
    _all_job_types,
]
