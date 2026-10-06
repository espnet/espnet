"""Shared validation for a class's field declarations."""

from espnet3.api.inference.check import check_declaration
from espnet3.api.inference.field import Field


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
