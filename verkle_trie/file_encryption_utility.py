
import os
import sys

def xor_cipher(data: bytes, key: bytes) -> bytes:
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def encrypt_file(input_path: str, output_path: str, key: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    with open(input_path, 'rb') as f:
        plaintext = f.read()
    
    key_bytes = key.encode('utf-8')
    ciphertext = xor_cipher(plaintext, key_bytes)
    
    with open(output_path, 'wb') as f:
        f.write(ciphertext)
    
    print(f"File encrypted successfully. Output: {output_path}")

def decrypt_file(input_path: str, output_path: str, key: str):
    encrypt_file(input_path, output_path, key)
    print(f"File decrypted successfully. Output: {output_path}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = sys.argv[4]
    
    if operation == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Error: Operation must be 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()