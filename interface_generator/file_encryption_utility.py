
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.password = password.encode('utf-8')
        self.salt = get_random_bytes(16)
        self.key = self._derive_key()

    def _derive_key(self):
        return PBKDF2(self.password, self.salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path + '.enc'

        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)

        with open(input_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

        with open(output_path, 'wb') as f:
            f.write(self.salt + iv + ciphertext)

        return output_path

    def decrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'

        with open(input_path, 'rb') as f:
            data = f.read()

        salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]

        key = PBKDF2(self.password, salt, dkLen=32, count=1000000)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        try:
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        except ValueError as e:
            raise ValueError("Decryption failed. Incorrect password or corrupted file.") from e

        with open(output_path, 'wb') as f:
            f.write(plaintext)

        return output_path

    def calculate_hash(self, file_path, algorithm='sha256'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        hash_func = getattr(hashlib, algorithm, None)
        if hash_func is None:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        with open(file_path, 'rb') as f:
            file_hash = hash_func()
            while chunk := f.read(8192):
                file_hash.update(chunk)

        return file_hash.hexdigest()

def example_usage():
    encryptor = FileEncryptor("secure_password_123")
    
    original_file = "test_document.txt"
    encrypted_file = encryptor.encrypt_file(original_file)
    print(f"Encrypted file saved as: {encrypted_file}")
    
    decrypted_file = encryptor.decrypt_file(encrypted_file)
    print(f"Decrypted file saved as: {decrypted_file}")
    
    original_hash = encryptor.calculate_hash(original_file)
    decrypted_hash = encryptor.calculate_hash(decrypted_file)
    
    if original_hash == decrypted_hash:
        print("File integrity verified: Hashes match")
    else:
        print("WARNING: File integrity check failed")

if __name__ == "__main__":
    example_usage()