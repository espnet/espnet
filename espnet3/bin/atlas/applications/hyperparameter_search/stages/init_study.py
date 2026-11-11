"""Initialize an Atlas study directory."""

from __future__ import annotations

from pathlib import Path

from espnet3.bin.atlas.core.models import StageResult
from espnet3.bin.atlas.core.stage import AtlasStage

PROGRAM_TEMPLATE = """# Atlas Objective

Describe:
- target metric
- constraints
- hypotheses to test
- forbidden changes
"""


class InitStudyStage(AtlasStage):
    """Create the stable study directory scaffold."""

    def run(self, context) -> StageResult:
        study_dir = context.study_dir
        for rel in [
            "knowledge",
            "trials",
            "node_runs",
        ]:
            (study_dir / rel).mkdir(parents=True, exist_ok=True)
        program_path = study_dir / "program.md"
        if not program_path.exists():
            program_path.write_text(PROGRAM_TEMPLATE, encoding="utf-8")
        for required in ["atlas.yaml", "graph.yaml"]:
            path = study_dir / required
            if not path.exists():
                return StageResult(
                    status="failure",
                    message=f"Missing required study file: {path}",
                )
        return StageResult(
            status="success",
            message="Study initialized",
            artifacts={"program": str(program_path)},
        )
