
from cryptography.fernet import Fernet
import os
import sys

class FileEncryptor:
    def __init__(self, key_file='secret.key'):
        self.key_file = key_file
        self.key = None
        self.cipher = None
        
    def generate_key(self):
        self.key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(self.key)
        print(f"Key generated and saved to {self.key_file}")
        return self.key
    
    def load_key(self):
        if not os.path.exists(self.key_file):
            raise FileNotFoundError(f"Key file {self.key_file} not found")
        
        with open(self.key_file, 'rb') as f:
            self.key = f.read()
        self.cipher = Fernet(self.key)
        return self.key
    
    def encrypt_file(self, input_file, output_file=None):
        if not self.cipher:
            self.load_key()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        if output_file is None:
            output_file = input_file + '.encrypted'
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted_data)
        
        print(f"File encrypted: {output_file}")
        return output_file
    
    def decrypt_file(self, input_file, output_file=None):
        if not self.cipher:
            self.load_key()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        if output_file is None:
            if input_file.endswith('.encrypted'):
                output_file = input_file[:-10]
            else:
                output_file = input_file + '.decrypted'
        
        with open(input_file, 'rb') as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        
        print(f"File decrypted: {output_file}")
        return output_file

def main():
    if len(sys.argv) < 3:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <filename>")
        print("       python file_encryption_utility.py generate-key")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    encryptor = FileEncryptor()
    
    if action == 'generate-key':
        encryptor.generate_key()
    elif action == 'encrypt':
        if len(sys.argv) < 3:
            print("Error: Filename required for encryption")
            sys.exit(1)
        filename = sys.argv[2]
        encryptor.encrypt_file(filename)
    elif action == 'decrypt':
        if len(sys.argv) < 3:
            print("Error: Filename required for decryption")
            sys.exit(1)
        filename = sys.argv[2]
        encryptor.decrypt_file(filename)
    else:
        print(f"Error: Unknown action '{action}'")
        print("Valid actions: generate-key, encrypt, decrypt")
        sys.exit(1)

if __name__ == '__main__':
    main()