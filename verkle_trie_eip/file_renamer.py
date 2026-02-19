
import os
import sys
from datetime import datetime

def add_timestamp_prefix(directory='.'):
    """
    Rename all files in the specified directory by adding a timestamp prefix.
    """
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{timestamp}_{filename}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        
        print(f"Total files renamed: {len(files)}")
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    add_timestamp_prefix(target_dir)