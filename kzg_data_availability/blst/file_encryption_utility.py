
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'rb') as f:
            plaintext = f.read()

        salt = os.urandom(self.salt_length)
        key = self._derive_key(salt)
        iv = os.urandom(16)

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'rb') as f:
            data = f.read()

        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]

        key = self._derive_key(salt)

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]

        with open(output_path, 'wb') as f:
            f.write(plaintext)

def generate_random_key() -> str:
    random_bytes = os.urandom(32)
    return base64.urlsafe_b64encode(random_bytes).decode()

if __name__ == "__main__":
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)
    
    encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")
    
    with open("test_decrypted.txt", "rb") as f:
        result = f.read()
    
    print("Original:", test_data)
    print("Decrypted:", result)
    print("Match:", test_data == result)
    
    for fname in ["test_plain.txt", "test_encrypted.bin", "test_decrypted.txt"]:
        if os.path.exists(fname):
            os.remove(fname)
    
    print("Random key:", generate_random_key())