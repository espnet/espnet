"""The old api.inference import paths still work, unchanged, through shims."""

import espnet3.api.inference as old
import espnet3.api.inference.kinds as old_kinds
import espnet3.components.contract as new
import espnet3.components.contract.kinds as new_kinds
from espnet3.components.contract.check import check_declaration
from espnet3.components.contract.field import Field


def test_field_is_the_same_object():
    assert old.Field is new.Field


def test_kinds_is_the_same_dict():
    assert old.KINDS is new.KINDS
    assert old_kinds.KINDS is new_kinds.KINDS


def test_kind_classes_are_the_same_objects():
    assert old.Kind is new.Kind
    assert old.AudioKind is new.AudioKind
    assert old.TextKind is new.TextKind
    assert old.SegmentsKind is new.SegmentsKind
    assert old.Audio is new.Audio
    assert old.register_kind is new.register_kind


def test_check_declaration_accepts_a_good_declaration():
    class Good:
        inputs = (Field("speech", "audio"), Field("prompt", "text", optional=True))
        outputs = (Field("text", "text"),)

    check_declaration(Good)


def test_check_declaration_rejects_empty():
    class Empty:
        inputs = ()
        outputs = (Field("text", "text"),)

    try:
        check_declaration(Empty)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_check_declaration_rejects_duplicate_names():
    class Dup:
        inputs = (Field("speech", "audio"), Field("speech", "text"))
        outputs = (Field("text", "text"),)

    try:
        check_declaration(Dup)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_check_declaration_rejects_required_after_optional():
    class Order:
        inputs = (Field("prompt", "text", optional=True), Field("speech", "audio"))
        outputs = (Field("text", "text"),)

    try:
        check_declaration(Order)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_check_declaration_rejects_optional_output():
    class OptOut:
        inputs = (Field("speech", "audio"),)
        outputs = (Field("text", "text", optional=True),)

    try:
        check_declaration(OptOut)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_check_declaration_honors_custom_attribute_names():
    class Renamed:
        in_fields = (Field("speech", "audio"),)
        out_fields = (Field("text", "text"),)

    check_declaration(Renamed, inputs_attr="in_fields", outputs_attr="out_fields")
