
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes) -> bytes:
        return PBKDF2(self.password, salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        salt = get_random_bytes(self.salt_length)
        key = self.derive_key(salt)

        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv

        with open(input_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'rb') as f:
            data = f.read()

        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]

        key = self.derive_key(salt)
        cipher = AES.new(key, AES.MODE_CBC, iv=iv)

        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

        with open(output_path, 'wb') as f:
            f.write(plaintext)

    def calculate_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message for encryption testing."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)
    
    encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")
    
    original_hash = encryptor.calculate_hash("test_plain.txt")
    decrypted_hash = encryptor.calculate_hash("test_decrypted.txt")
    
    print(f"Original file hash: {original_hash}")
    print(f"Decrypted file hash: {decrypted_hash}")
    
    if original_hash == decrypted_hash:
        print("Encryption/decryption successful: Hashes match")
    else:
        print("Error: Hashes do not match")
    
    for file in ["test_plain.txt", "test_encrypted.bin", "test_decrypted.txt"]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()