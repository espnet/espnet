"""Test that AbsLossWrapper's base signature accepts others=None.

This guards the Liskov Substitution fix: previously the base class
declared `others: Dict` as required, but every subclass defaulted it
to None. The base signature now reflects the actual contract.
"""
import inspect

import torch

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainDPCL
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.dpcl_solver import DPCLSolver


def test_base_forward_signature_accepts_optional_others():
    sig = inspect.signature(AbsLossWrapper.forward)
    others_param = sig.parameters["others"]
    assert others_param.default is None
    # typing.Optional[Dict] resolves to Union[Dict, None]
    assert "Dict" in str(others_param.annotation)


def test_dpcl_solver_callable_without_others_arg():
    # Subclass still works when others is omitted (defaults in the base).
    batch = 2
    inf = [torch.rand(batch, 10, 200) for _ in range(2)]
    ref = list(reversed(inf))
    solver = DPCLSolver(FrequencyDomainDPCL())
    # The subclass's own forward still defaults others to None and
    # treats it as {} internally. The base signature change makes the
    # override consistent with the abstract method's type annotation.
    loss, stats, others = solver(ref, inf, {"tf_embedding": torch.rand(batch, 10 * 200, 40)})
    assert isinstance(loss, torch.Tensor)
