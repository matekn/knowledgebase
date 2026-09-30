import hashlib

def hash_password(password: str, salt: bytes) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return dk.hex()


def verify_password(password: str, salt: bytes, expected: str) -> bool:
    return hash_password(password, salt) == expected
