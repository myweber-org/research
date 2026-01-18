
import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, key=None):
        self.key = key or self.generate_key()
    
    @staticmethod
    def generate_key():
        return get_random_bytes(32)
    
    def derive_key(self, password, salt):
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, dklen=32)
    
    def encrypt_file(self, input_path, output_path=None, password=None):
        if password:
            salt = get_random_bytes(16)
            key = self.derive_key(password, salt)
        else:
            salt = None
            key = self.key
        
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        output_path = output_path or input_path + '.enc'
        with open(output_path, 'wb') as f:
            if salt:
                f.write(salt)
            f.write(iv)
            f.write(ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path, output_path=None, password=None):
        with open(input_path, 'rb') as f:
            if password:
                salt = f.read(16)
                key = self.derive_key(password, salt)
            else:
                salt = None
                key = self.key
            
            iv = f.read(16)
            ciphertext = f.read()
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        output_path = output_path or input_path.replace('.enc', '')
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path
    
    def save_key(self, path):
        with open(path, 'wb') as f:
            f.write(self.key)
    
    @classmethod
    def load_key(cls, path):
        with open(path, 'rb') as f:
            key = f.read()
        return cls(key)

def main():
    encryptor = FileEncryptor()
    
    test_data = b"Test data for encryption and decryption."
    with open('test.txt', 'wb') as f:
        f.write(test_data)
    
    encrypted = encryptor.encrypt_file('test.txt', password='securepass')
    print(f"Encrypted file: {encrypted}")
    
    decryptor = FileEncryptor()
    decrypted = decryptor.decrypt_file(encrypted, password='securepass')
    print(f"Decrypted file: {decrypted}")
    
    with open(decrypted, 'rb') as f:
        result = f.read()
    
    if result == test_data:
        print("Encryption/decryption successful")
    else:
        print("Encryption/decryption failed")
    
    os.remove('test.txt')
    os.remove(encrypted)
    os.remove(decrypted)

if __name__ == '__main__':
    main()