# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import binascii
import os

import pytest
from cryptography.hazmat.primitives.ciphers import algorithms, base, modes

from ...utils import load_nist_vectors
from .utils import generate_aead_test


@pytest.mark.supported(
    only_if=lambda backend: backend.cipher_supported(
        algorithms.AES(b"\x00" * 16), modes.GCM(b"\x00" * 12)
    ),
    skip_message="Does not support AES GCM",
)
class TestAESModeGCM:
    test_gcm = generate_aead_test(
        load_nist_vectors,
        os.path.join("ciphers", "AES", "GCM"),
        [
            "gcmDecrypt128.rsp",
            "gcmDecrypt192.rsp",
            "gcmDecrypt256.rsp",
            "gcmEncryptExtIV128.rsp",
            "gcmEncryptExtIV192.rsp",
            "gcmEncryptExtIV256.rsp",
        ],
        algorithms.AES,
        modes.GCM,
    )

    def test_gcm_tag_with_only_aad(self, backend):
        key = binascii.unhexlify(b"5211242698bed4774a090620a6ca56f3")
        iv = binascii.unhexlify(b"b1e1349120b6e832ef976f5d")
        aad = binascii.unhexlify(b"b6d729aab8e6416d7002b9faa794c410d8d2f193")
        tag = binascii.unhexlify(b"0f247e7f9c2505de374006738018493b")

        cipher = base.Cipher(algorithms.AES(key), modes.GCM(iv), backend=backend)
        encryptor = cipher.encryptor()
        encryptor.authenticate_additional_data(aad)
        encryptor.finalize()
        assert encryptor.tag == tag

    def test_gcm_ciphertext_with_no_aad(self, backend):
        key = binascii.unhexlify(b"e98b72a9881a84ca6b76e0f43e68647a")
        iv = binascii.unhexlify(b"8b23299fde174053f3d652ba")
        ct = binascii.unhexlify(b"5a3c1cf1985dbb8bed818036fdd5ab42")
        tag = binascii.unhexlify(b"23c7ab0f952b7091cd324835043b5eb5")
        pt = binascii.unhexlify(b"28286a321293253c3e0aa2704a278032")

        cipher = base.Cipher(algorithms.AES(key), modes.GCM(iv), backend=backend)
        encryptor = cipher.encryptor()
        computed_ct = encryptor.update(pt) + encryptor.finalize()
        assert computed_ct == ct
        assert encryptor.tag == tag

    def test_gcm_ciphertext_limit(self, backend):
        encryptor = base.Cipher(
            algorithms.AES(b"\x00" * 16),
            modes.GCM(b"\x01" * 16),
            backend=backend,
        ).encryptor()
        new_max = modes.GCM._MAX_ENCRYPTED_BYTES - 16
        encryptor._bytes_processed = new_max  # type: ignore[attr-defined]
        encryptor.update(b"0" * 16)
        max = modes.GCM._MAX_ENCRYPTED_BYTES
        assert encryptor._bytes_processed == max  # type: ignore[attr-defined]
        with pytest.raises(ValueError):
            encryptor.update(b"0")

    def test_gcm_aad_limit(self, backend):
        encryptor = base.Cipher(
            algorithms.AES(b"\x00" * 16),
            modes.GCM(b"\x01" * 16),
            backend=backend,
        ).encryptor()
        new_max = modes.GCM._MAX_AAD_BYTES - 16
        encryptor._aad_bytes_processed = new_max  # type: ignore[attr-defined]
        encryptor.authenticate_additional_data(b"0" * 16)
        max = modes.GCM._MAX_AAD_BYTES
        assert encryptor._aad_bytes_processed == max  # type: ignore[attr-defined]
        with pytest.raises(ValueError):
            encryptor.authenticate_additional_data(b"0")

    def test_gcm_ciphertext_increments(self, backend):
        encryptor = base.Cipher(
            algorithms.AES(b"\x00" * 16),
            modes.GCM(b"\x01" * 16),
            backend=backend,
        ).encryptor()
        encryptor.update(b"0" * 8)
        assert encryptor._bytes_processed == 8  # type: ignore[attr-defined]
        encryptor.update(b"0" * 7)
        assert encryptor._bytes_processed == 15  # type: ignore[attr-defined]
        encryptor.update(b"0" * 18)
        assert encryptor._bytes_processed == 33  # type: ignore[attr-defined]

    def test_gcm_aad_increments(self, backend):
        encryptor = base.Cipher(
            algorithms.AES(b"\x00" * 16),
            modes.GCM(b"\x01" * 16),
            backend=backend,
        ).encryptor()
        encryptor.authenticate_additional_data(b"0" * 8)
        assert encryptor._aad_bytes_processed == 8  # type: ignore[attr-defined]
        encryptor.authenticate_additional_data(b"0" * 18)
        assert encryptor._aad_bytes_processed == 26  # type: ignore[attr-defined]

    def test_gcm_tag_decrypt_none(self, backend):
        key = binascii.unhexlify(b"5211242698bed4774a090620a6ca56f3")
        iv = binascii.unhexlify(b"b1e1349120b6e832ef976f5d")
        aad = binascii.unhexlify(b"b6d729aab8e6416d7002b9faa794c410d8d2f193")

        encryptor = base.Cipher(
            algorithms.AES(key), modes.GCM(iv), backend=backend
        ).encryptor()
        encryptor.authenticate_additional_data(aad)
        encryptor.finalize()

        decryptor = base.Cipher(
            algorithms.AES(key), modes.GCM(iv), backend=backend
        ).decryptor()
        decryptor.authenticate_additional_data(aad)
        with pytest.raises(ValueError):
            decryptor.finalize()

    def test_gcm_tag_decrypt_mode(self, backend):
        key = binascii.unhexlify(b"5211242698bed4774a090620a6ca56f3")
        iv = binascii.unhexlify(b"b1e1349120b6e832ef976f5d")
        aad = binascii.unhexlify(b"b6d729aab8e6416d7002b9faa794c410d8d2f193")

        encryptor = base.Cipher(
            algorithms.AES(key), modes.GCM(iv), backend=backend
        ).encryptor()
        encryptor.authenticate_additional_data(aad)
        encryptor.finalize()
        tag = encryptor.tag

        decryptor = base.Cipher(
            algorithms.AES(key), modes.GCM(iv, tag), backend=backend
        ).decryptor()
        decryptor.authenticate_additional_data(aad)
        decryptor.finalize()

    def test_gcm_tag_decrypt_finalize(self, backend):
        key = binascii.unhexlify(b"5211242698bed4774a090620a6ca56f3")
        iv = binascii.unhexlify(b"b1e1349120b6e832ef976f5d")
        aad = binascii.unhexlify(b"b6d729aab8e6416d7002b9faa794c410d8d2f193")

        encryptor = base.Cipher(
            algorithms.AES(key), modes.GCM(iv), backend=backend
        ).encryptor()
        encryptor.authenticate_additional_data(aad)
        encryptor.finalize()
        tag = encryptor.tag

        decryptor = base.Cipher(
            algorithms.AES(key), modes.GCM(iv), backend=backend
        ).decryptor()
        decryptor.authenticate_additional_data(aad)

        decryptor.finalize_with_tag(tag)

    @pytest.mark.parametrize("tag", [b"tagtooshort", b"toolong" * 12])
    def test_gcm_tag_decrypt_finalize_tag_length(self, tag, backend):
        decryptor = base.Cipher(
            algorithms.AES(b"0" * 16), modes.GCM(b"0" * 12), backend=backend
        ).decryptor()
        with pytest.raises(ValueError):
            decryptor.finalize_with_tag(tag)

    def test_buffer_protocol(self, backend):
        data = bytearray(b"helloworld")
        enc = base.Cipher(
            algorithms.AES(bytearray(b"\x00" * 16)),
            modes.GCM(bytearray(b"\x00" * 12)),
            backend,
        ).encryptor()
        enc.authenticate_additional_data(bytearray(b"foo"))
        ct = enc.update(data) + enc.finalize()
        dec = base.Cipher(
            algorithms.AES(bytearray(b"\x00" * 16)),
            modes.GCM(bytearray(b"\x00" * 12), enc.tag),
            backend,
        ).decryptor()
        dec.authenticate_additional_data(bytearray(b"foo"))
        pt = dec.update(ct) + dec.finalize()
        assert pt == data

    @pytest.mark.parametrize("size", [8, 128])
    def test_gcm_min_max_iv(self, size, backend):
        if backend._fips_enabled:
            # Red Hat disables non-96-bit IV support as part of its FIPS
            # patches.
            pytest.skip("Non-96-bit IVs unsupported in FIPS mode.")

        key = os.urandom(16)
        iv = b"\x00" * size

        payload = b"data"
        encryptor = base.Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
        ct = encryptor.update(payload)
        encryptor.finalize()
        tag = encryptor.tag

        decryptor = base.Cipher(algorithms.AES(key), modes.GCM(iv)).decryptor()
        pt = decryptor.update(ct)

        decryptor.finalize_with_tag(tag)
        assert pt == payload

    @pytest.mark.parametrize("alg", [algorithms.AES128, algorithms.AES256])
    def test_alternate_aes_classes(self, alg, backend):
        data = bytearray(b"sixteen_byte_msg")
        cipher = base.Cipher(
            alg(b"0" * (alg.key_size // 8)), modes.GCM(b"\x00" * 12), backend
        )
        enc = cipher.encryptor()
        ct = enc.update(data) + enc.finalize()
        dec = cipher.decryptor()
        pt = dec.update(ct) + dec.finalize_with_tag(enc.tag)
        assert pt == data
