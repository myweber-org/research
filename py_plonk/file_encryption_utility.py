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
        with open(self.key_file, 'wb') as key_file:
            key_file.write(self.key)
        print(f"Key generated and saved to {self.key_file}")
        return self.key
    
    def load_key(self):
        if not os.path.exists(self.key_file):
            raise FileNotFoundError(f"Key file {self.key_file} not found")
        
        with open(self.key_file, 'rb') as key_file:
            self.key = key_file.read()
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
            raise ValueError(f"Decryption failed: {str(e)}")
        
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
        
        decrypted_text = self.cipher.decrypt(encrypted_text)
        return decrypted_text.decode()

def main():
    if len(sys.argv) < 3:
        print("Usage: python file_encryption_utility.py <command> <file> [output_file]")
        print("Commands: encrypt, decrypt, genkey")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    encryptor = FileEncryptor()
    
    try:
        if command == 'genkey':
            encryptor.generate_key()
            print("Key generation completed successfully")
        
        elif command == 'encrypt':
            if len(sys.argv) < 3:
                print("Error: Input file required for encryption")
                sys.exit(1)
            
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            encryptor.encrypt_file(input_file, output_file)
        
        elif command == 'decrypt':
            if len(sys.argv) < 3:
                print("Error: Input file required for decryption")
                sys.exit(1)
            
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            encryptor.decrypt_file(input_file, output_file)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: genkey, encrypt, decrypt")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()