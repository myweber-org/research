
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.password = password.encode()
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
        except ValueError:
            raise ValueError("Decryption failed. Incorrect password or corrupted file.")

        with open(output_path, 'wb') as f:
            f.write(plaintext)

        return output_path

def calculate_file_hash(file_path, algorithm='sha256'):
    hash_func = getattr(hashlib, algorithm)()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def main():
    import sys
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <file_path> <password>")
        sys.exit(1)

    action = sys.argv[1]
    file_path = sys.argv[2]
    password = sys.argv[3]

    encryptor = FileEncryptor(password)

    try:
        if action == 'encrypt':
            output = encryptor.encrypt_file(file_path)
            print(f"File encrypted successfully: {output}")
            print(f"SHA256 hash: {calculate_file_hash(output)}")
        elif action == 'decrypt':
            output = encryptor.decrypt_file(file_path)
            print(f"File decrypted successfully: {output}")
            print(f"SHA256 hash: {calculate_file_hash(output)}")
        else:
            print("Invalid action. Use 'encrypt' or 'decrypt'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()