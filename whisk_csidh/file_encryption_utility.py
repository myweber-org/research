
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _crypt(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt(self, plaintext: str) -> bytes:
        return self._crypt(plaintext.encode('utf-8'))
    
    def decrypt(self, ciphertext: bytes) -> str:
        return self._crypt(ciphertext).decode('utf-8')

def encrypt_file(input_path: str, output_path: str, key: str):
    cipher = XORCipher(key)
    
    with open(input_path, 'rb') as f:
        data = f.read()
    
    encrypted = cipher._crypt(data)
    
    with open(output_path, 'wb') as f:
        f.write(encrypted)
    
    print(f"File encrypted: {output_path}")

def decrypt_file(input_path: str, output_path: str, key: str):
    cipher = XORCipher(key)
    
    with open(input_path, 'rb') as f:
        data = f.read()
    
    decrypted = cipher._crypt(data)
    
    with open(output_path, 'wb') as f:
        f.write(decrypted)
    
    print(f"File decrypted: {output_path}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = sys.argv[4]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    if operation == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print(f"Error: Unknown operation '{operation}'. Use 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()