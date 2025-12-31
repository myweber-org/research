
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecureFileEncryptor:
    def __init__(self, salt_length=16, iterations=100000):
        self.salt_length = salt_length
        self.iterations = iterations
        self.backend = default_backend()

    def derive_key(self, password, salt):
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return kdf.derive(password.encode())

    def encrypt_file(self, input_path, output_path, password):
        salt = os.urandom(self.salt_length)
        key = self.derive_key(password, salt)
        
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as f_in:
            plaintext = f_in.read()
        
        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        with open(output_path, 'wb') as f_out:
            f_out.write(salt + iv + ciphertext)
        
        return True

    def decrypt_file(self, input_path, output_path, password):
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]
        
        key = self.derive_key(password, salt)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]
        
        with open(output_path, 'wb') as f_out:
            f_out.write(plaintext)
        
        return True

def generate_secure_password(length=32):
    return base64.urlsafe_b64encode(os.urandom(length)).decode()[:length]

if __name__ == "__main__":
    encryptor = SecureFileEncryptor()
    
    test_data = b"This is a secret message that needs encryption."
    with open('test_input.txt', 'wb') as f:
        f.write(test_data)
    
    password = generate_secure_password()
    print(f"Generated password: {password}")
    
    encryptor.encrypt_file('test_input.txt', 'encrypted.dat', password)
    encryptor.decrypt_file('encrypted.dat', 'decrypted.txt', password)
    
    with open('decrypted.txt', 'rb') as f:
        result = f.read()
    
    if result == test_data:
        print("Encryption/decryption successful")
    else:
        print("Encryption/decryption failed")
    
    os.remove('test_input.txt')
    os.remove('encrypted.dat')
    os.remove('decrypted.txt')