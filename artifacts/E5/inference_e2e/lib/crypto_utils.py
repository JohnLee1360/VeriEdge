from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_task_key() -> bytes:
    return os.urandom(32)


def encrypt_bytes_aes_gcm(plaintext: bytes, key: bytes) -> dict[str, str]:
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return {
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_bytes_aes_gcm(ciphertext_b64: str, nonce_b64: str, key: bytes) -> bytes:
    ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"))
    nonce = base64.b64decode(nonce_b64.encode("ascii"))
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def load_public_key(path: Path):
    return serialization.load_pem_public_key(path.read_bytes())


def load_private_key(path: Path):
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def encrypt_task_key_for_provider(task_key: bytes, public_key_path: Path) -> str:
    public_key = load_public_key(public_key_path)
    encrypted = public_key.encrypt(
        task_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return base64.b64encode(encrypted).decode("ascii")


def decrypt_task_key_from_request(encrypted_key_b64: str, private_key_path: Path) -> bytes:
    private_key = load_private_key(private_key_path)
    encrypted = base64.b64decode(encrypted_key_b64.encode("ascii"))
    return private_key.decrypt(
        encrypted,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )


def generate_rsa_keypair(private_key_path: Path, public_key_path: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
