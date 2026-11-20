from espnet2.lid.loss.linear_softmax import LinearSoftmax
from espnet2.spk.encoder.e_branchformer_encoder import MfaEBranchformerEncoder
from espnet3.systems.esp2_lid.task import LIDTask


def _choices(name):
    (choices,) = [c for c in LIDTask.class_choices_list if c.name == name]
    return choices


def test_mfa_e_branchformer_and_linear_softmax_are_registered():
    assert _choices("encoder").get_class("mfa_e_branchformer") is (
        MfaEBranchformerEncoder
    )
    assert _choices("loss").get_class("linear_softmax") is LinearSoftmax
