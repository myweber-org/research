
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecureFileEncryptor:
    def __init__(self, salt_length=16, iterations=100000):
        self.salt_length = salt_length
        self.iterations = iterations
        self.backend = default_backend()

    def derive_key(self, password, salt):
        kdf = PBKDF2HMAC(
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
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)
        
        return True

    def decrypt_file(self, input_path, output_path, password):
        with open(input_path, 'rb') as f:
            data = f.read()
        
        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length+16]
        ciphertext = data[self.salt_length+16:]
        
        key = self.derive_key(password, salt)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return True

def main():
    encryptor = SecureFileEncryptor()
    
    test_data = b"This is a secret message that needs encryption."
    with open('test_input.txt', 'wb') as f:
        f.write(test_data)
    
    password = "StrongPassword123!"
    
    encryptor.encrypt_file('test_input.txt', 'encrypted.dat', password)
    print("File encrypted successfully")
    
    encryptor.decrypt_file('encrypted.dat', 'decrypted.txt', password)
    print("File decrypted successfully")
    
    with open('decrypted.txt', 'rb') as f:
        decrypted_data = f.read()
    
    if decrypted_data == test_data:
        print("Encryption/decryption verified successfully")
    
    os.remove('test_input.txt')
    os.remove('encrypted.dat')
    os.remove('decrypted.txt')

if __name__ == "__main__":
    main()