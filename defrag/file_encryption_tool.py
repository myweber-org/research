
import os
import sys

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted_data = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)
    
    mode = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        key = int(sys.argv[4])
        if not 0 <= key <= 255:
            raise ValueError
    except ValueError:
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist")
        sys.exit(1)
    
    if mode == 'encrypt':
        if encrypt_file(input_file, output_file, key):
            print(f"File encrypted successfully: {output_file}")
        else:
            print("Encryption failed")
    elif mode == 'decrypt':
        if decrypt_file(input_file, output_file, key):
            print(f"File decrypted successfully: {output_file}")
        else:
            print("Decryption failed")
    else:
        print("Mode must be 'encrypt' or 'decrypt'")

if __name__ == "__main__":
    main()