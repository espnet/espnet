"""Tests for check_budget."""

from __future__ import annotations

from types import SimpleNamespace


def test_check_budget_routes_done_only_when_the_budget_is_spent():
    from espnet3.bin.atlas.stages.check_budget import CheckBudgetStage

    for spent, expected in ((False, "continue"), (True, "done")):
        context = SimpleNamespace(scheduler=SimpleNamespace(should_stop=lambda: spent))
        assert CheckBudgetStage().run(context).status == expected
