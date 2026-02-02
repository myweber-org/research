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
            file_data = f.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
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
            print(f"Decryption failed: {e}")
            return None
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        
        print(f"File decrypted: {output_file}")
        return output_file
    
    def encrypt_string(self, text):
        if not self.cipher:
            self.load_key()
        
        if isinstance(text, str):
            text = text.encode()
        
        encrypted_text = self.cipher.encrypt(text)
        return encrypted_text.decode()
    
    def decrypt_string(self, encrypted_text):
        if not self.cipher:
            self.load_key()
        
        if isinstance(encrypted_text, str):
            encrypted_text = encrypted_text.encode()
        
        try:
            decrypted_text = self.cipher.decrypt(encrypted_text)
            return decrypted_text.decode()
        except Exception as e:
            print(f"Decryption failed: {e}")
            return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_encryption_utility.py <command> [options]")
        print("Commands: generate-key, encrypt <file>, decrypt <file>")
        return
    
    command = sys.argv[1]
    encryptor = FileEncryptor()
    
    try:
        if command == 'generate-key':
            encryptor.generate_key()
        
        elif command == 'encrypt':
            if len(sys.argv) < 3:
                print("Please specify file to encrypt")
                return
            
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            encryptor.encrypt_file(input_file, output_file)
        
        elif command == 'decrypt':
            if len(sys.argv) < 3:
                print("Please specify file to decrypt")
                return
            
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            encryptor.decrypt_file(input_file, output_file)
        
        else:
            print(f"Unknown command: {command}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()