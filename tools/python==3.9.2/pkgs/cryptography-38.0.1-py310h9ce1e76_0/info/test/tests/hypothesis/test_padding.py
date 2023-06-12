# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from cryptography.hazmat.primitives.padding import ANSIX923, PKCS7
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import binary, integers


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(integers(min_value=1, max_value=255), binary())
def test_pkcs7(block_size, data):
    # Generate in [1, 31] so we can easily get block_size in bits by
    # multiplying by 8.
    p = PKCS7(block_size=block_size * 8)
    padder = p.padder()
    unpadder = p.unpadder()

    padded = padder.update(data) + padder.finalize()

    assert unpadder.update(padded) + unpadder.finalize() == data


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(integers(min_value=1, max_value=255), binary())
def test_ansix923(block_size, data):
    a = ANSIX923(block_size=block_size * 8)
    padder = a.padder()
    unpadder = a.unpadder()

    padded = padder.update(data) + padder.finalize()

    assert unpadder.update(padded) + unpadder.finalize() == data
