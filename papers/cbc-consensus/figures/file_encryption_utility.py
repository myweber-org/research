import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.key = hashlib.sha256(password.encode()).digest()
    
    def encrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        if output_path is None:
            output_path = input_path + '.enc'
        
        with open(output_path, 'wb') as f:
            f.write(iv + ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path
    
    def encrypt_string(self, plaintext):
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
        return b64encode(iv + ciphertext).decode()
    
    def decrypt_string(self, encrypted_text):
        data = b64decode(encrypted_text.encode())
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return plaintext.decode()

def example_usage():
    encryptor = FileEncryptor("my_secure_password")
    
    # Encrypt a string
    encrypted = encryptor.encrypt_string("Secret message")
    print(f"Encrypted: {encrypted}")
    
    # Decrypt the string
    decrypted = encryptor.decrypt_string(encrypted)
    print(f"Decrypted: {decrypted}")
    
    # Create a test file
    test_content = b"This is confidential data that needs protection."
    with open('test_file.txt', 'wb') as f:
        f.write(test_content)
    
    # Encrypt the file
    encrypted_file = encryptor.encrypt_file('test_file.txt')
    print(f"Encrypted file created: {encrypted_file}")
    
    # Decrypt the file
    decrypted_file = encryptor.decrypt_file(encrypted_file)
    print(f"Decrypted file created: {decrypted_file}")
    
    # Cleanup
    os.remove('test_file.txt')
    os.remove(encrypted_file)
    os.remove(decrypted_file)

if __name__ == "__main__":
    example_usage()