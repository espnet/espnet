# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import binascii
import os

import pytest
from cryptography.exceptions import InvalidSignature, _Reasons
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ...utils import load_vectors_from_file, raises_unsupported_algorithm


def load_ed25519_vectors(vector_data):
    """
    djb's ed25519 vectors are structured as a colon delimited array:
        0: secret key (32 bytes) + public key (32 bytes)
        1: public key (32 bytes)
        2: message (0+ bytes)
        3: signature + message (64+ bytes)
    """
    data = []
    for line in vector_data:
        secret_key, public_key, message, signature, _ = line.split(":")
        secret_key = secret_key[0:64]
        signature = signature[0:128]
        data.append(
            {
                "secret_key": secret_key,
                "public_key": public_key,
                "message": message,
                "signature": signature,
            }
        )
    return data


@pytest.mark.supported(
    only_if=lambda backend: not backend.ed25519_supported(),
    skip_message="Requires OpenSSL without Ed25519 support",
)
def test_ed25519_unsupported(backend):
    with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM):
        Ed25519PublicKey.from_public_bytes(b"0" * 32)

    with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM):
        Ed25519PrivateKey.from_private_bytes(b"0" * 32)

    with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM):
        Ed25519PrivateKey.generate()


@pytest.mark.supported(
    only_if=lambda backend: backend.ed25519_supported(),
    skip_message="Requires OpenSSL with Ed25519 support",
)
class TestEd25519Signing:
    def test_sign_verify_input(self, backend, subtests):
        vectors = load_vectors_from_file(
            os.path.join("asymmetric", "Ed25519", "sign.input"),
            load_ed25519_vectors,
        )
        for vector in vectors:
            with subtests.test():
                sk = binascii.unhexlify(vector["secret_key"])
                pk = binascii.unhexlify(vector["public_key"])
                message = binascii.unhexlify(vector["message"])
                signature = binascii.unhexlify(vector["signature"])
                private_key = Ed25519PrivateKey.from_private_bytes(sk)
                computed_sig = private_key.sign(message)
                assert computed_sig == signature
                public_key = private_key.public_key()
                assert (
                    public_key.public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                    == pk
                )
                public_key.verify(signature, message)

    def test_invalid_signature(self, backend):
        key = Ed25519PrivateKey.generate()
        signature = key.sign(b"test data")
        with pytest.raises(InvalidSignature):
            key.public_key().verify(signature, b"wrong data")

        with pytest.raises(InvalidSignature):
            key.public_key().verify(b"0" * 64, b"test data")

    def test_generate(self, backend):
        key = Ed25519PrivateKey.generate()
        assert key
        assert key.public_key()

    def test_load_public_bytes(self, backend):
        public_key = Ed25519PrivateKey.generate().public_key()
        public_bytes = public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        public_key2 = Ed25519PublicKey.from_public_bytes(public_bytes)
        assert public_bytes == public_key2.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def test_invalid_type_public_bytes(self, backend):
        with pytest.raises(TypeError):
            Ed25519PublicKey.from_public_bytes(object())  # type: ignore[arg-type]

    def test_invalid_type_private_bytes(self, backend):
        with pytest.raises(TypeError):
            Ed25519PrivateKey.from_private_bytes(object())  # type: ignore[arg-type]

    def test_invalid_length_from_public_bytes(self, backend):
        with pytest.raises(ValueError):
            Ed25519PublicKey.from_public_bytes(b"a" * 31)
        with pytest.raises(ValueError):
            Ed25519PublicKey.from_public_bytes(b"a" * 33)

    def test_invalid_length_from_private_bytes(self, backend):
        with pytest.raises(ValueError):
            Ed25519PrivateKey.from_private_bytes(b"a" * 31)
        with pytest.raises(ValueError):
            Ed25519PrivateKey.from_private_bytes(b"a" * 33)

    def test_invalid_private_bytes(self, backend):
        key = Ed25519PrivateKey.generate()
        with pytest.raises(ValueError):
            key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                None,  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError):
            key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.PKCS8,
                None,  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError):
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )

    def test_invalid_public_bytes(self, backend):
        key = Ed25519PrivateKey.generate().public_key()
        with pytest.raises(ValueError):
            key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        with pytest.raises(ValueError):
            key.public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.PKCS1
            )

        with pytest.raises(ValueError):
            key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.Raw)

    @pytest.mark.parametrize(
        ("encoding", "fmt", "encryption", "passwd", "load_func"),
        [
            (
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"password"),
                b"password",
                serialization.load_pem_private_key,
            ),
            (
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"password"),
                b"password",
                serialization.load_der_private_key,
            ),
            (
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
                None,
                serialization.load_pem_private_key,
            ),
            (
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
                None,
                serialization.load_der_private_key,
            ),
        ],
    )
    def test_round_trip_private_serialization(
        self, encoding, fmt, encryption, passwd, load_func, backend
    ):
        key = Ed25519PrivateKey.generate()
        serialized = key.private_bytes(encoding, fmt, encryption)
        loaded_key = load_func(serialized, passwd, backend)
        assert isinstance(loaded_key, Ed25519PrivateKey)

    def test_buffer_protocol(self, backend):
        private_bytes = os.urandom(32)
        key = Ed25519PrivateKey.from_private_bytes(bytearray(private_bytes))
        assert (
            key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            == private_bytes
        )
