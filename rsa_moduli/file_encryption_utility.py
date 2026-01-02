
import os
import hashlib
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

    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        if output_path is None:
            output_path = input_path + '.enc'

        salt = get_random_bytes(self.salt_length)
        key = self.derive_key(salt)

        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv

        with open(input_path, 'rb') as f_in:
            plaintext = f_in.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

        with open(output_path, 'wb') as f_out:
            f_out.write(salt + iv + ciphertext)

        return output_path

    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'

        with open(input_path, 'rb') as f_in:
            data = f_in.read()

        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]

        key = self.derive_key(salt)
        cipher = AES.new(key, AES.MODE_CBC, iv=iv)

        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

        with open(output_path, 'wb') as f_out:
            f_out.write(plaintext)

        return output_path

    def calculate_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

def example_usage():
    encryptor = FileEncryptor('secure_password_123')
    
    original_file = 'test_document.txt'
    encrypted_file = encryptor.encrypt_file(original_file)
    print(f'Encrypted: {encrypted_file}')
    
    decrypted_file = encryptor.decrypt_file(encrypted_file)
    print(f'Decrypted: {decrypted_file}')
    
    original_hash = encryptor.calculate_hash(original_file)
    decrypted_hash = encryptor.calculate_hash(decrypted_file)
    
    if original_hash == decrypted_hash:
        print('Hash verification: SUCCESS')
    else:
        print('Hash verification: FAILED')

if __name__ == '__main__':
    example_usage()