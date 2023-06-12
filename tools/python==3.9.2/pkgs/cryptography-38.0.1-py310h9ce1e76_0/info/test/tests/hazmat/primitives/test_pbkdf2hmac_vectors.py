# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import pytest
from cryptography.hazmat.primitives import hashes

from ...utils import load_nist_vectors
from .utils import generate_pbkdf2_test


@pytest.mark.supported(
    only_if=lambda backend: backend.pbkdf2_hmac_supported(hashes.SHA1()),
    skip_message="Does not support SHA1 for PBKDF2HMAC",
)
class TestPBKDF2HMACSHA1:
    test_pbkdf2_sha1 = generate_pbkdf2_test(
        load_nist_vectors,
        "KDF",
        ["rfc-6070-PBKDF2-SHA1.txt"],
        hashes.SHA1(),
    )
