import os
from cryptography.fernet import Fernet

def generate_key():
    return Fernet.generate_key()

def save_key(key, key_file='secret.key'):
    with open(key_file, 'wb') as f:
        f.write(key)

def load_key(key_file='secret.key'):
    return open(key_file, 'rb').read()

def encrypt_file(file_path, key):
    fernet = Fernet(key)
    with open(file_path, 'rb') as f:
        original_data = f.read()
    encrypted_data = fernet.encrypt(original_data)
    with open(file_path + '.encrypted', 'wb') as f:
        f.write(encrypted_data)
    return file_path + '.encrypted'

def decrypt_file(encrypted_file_path, key):
    fernet = Fernet(key)
    with open(encrypted_file_path, 'rb') as f:
        encrypted_data = f.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    original_file_path = encrypted_file_path.replace('.encrypted', '.decrypted')
    with open(original_file_path, 'wb') as f:
        f.write(decrypted_data)
    return original_file_path

def main():
    key = generate_key()
    save_key(key)
    
    sample_file = 'sample.txt'
    with open(sample_file, 'w') as f:
        f.write('This is a secret message.')
    
    encrypted_file = encrypt_file(sample_file, key)
    print(f'File encrypted: {encrypted_file}')
    
    decrypted_file = decrypt_file(encrypted_file, key)
    print(f'File decrypted: {decrypted_file}')
    
    with open(decrypted_file, 'r') as f:
        print(f'Decrypted content: {f.read()}')
    
    os.remove(sample_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    os.remove('secret.key')

if __name__ == '__main__':
    main()