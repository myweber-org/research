
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory, extensions=None, days_old=7):
    """
    Remove temporary files from a directory based on extension and age.
    
    Args:
        directory (str or Path): Directory to clean.
        extensions (list, optional): List of file extensions to target.
            Defaults to common temporary extensions.
        days_old (int): Remove files older than this many days.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache', '.bak']
    
    target_dir = Path(directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    if not target_dir.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    removed_count = 0
    total_size = 0
    
    for item in target_dir.rglob('*'):
        if item.is_file():
            file_ext = item.suffix.lower()
            
            if file_ext in extensions or item.name.startswith('~'):
                file_age = item.stat().st_mtime
                
                if file_age < cutoff_time:
                    try:
                        file_size = item.stat().st_size
                        item.unlink()
                        removed_count += 1
                        total_size += file_size
                        print(f"Removed: {item.name} ({file_size} bytes)")
                    except (PermissionError, OSError) as e:
                        print(f"Failed to remove {item.name}: {e}")
    
    print(f"\nCleaning complete:")
    print(f"  Files removed: {removed_count}")
    print(f"  Total space freed: {total_size} bytes")
    print(f"  Target directory: {target_dir.absolute()}")

def create_sample_temp_files(directory, count=5):
    """
    Create sample temporary files for testing.
    
    Args:
        directory (str or Path): Directory to create files in.
        count (int): Number of files to create.
    """
    target_dir = Path(directory)
    target_dir.mkdir(exist_ok=True)
    
    extensions = ['.tmp', '.temp', '.log', '.cache']
    
    for i in range(count):
        ext = extensions[i % len(extensions)]
        temp_file = target_dir / f"sample_file_{i}{ext}"
        
        with open(temp_file, 'w') as f:
            f.write(f"This is a sample temporary file #{i}\n")
            f.write("Created for testing cleanup functionality.\n")
        
        # Make file older by modifying its timestamp
        old_time = time.time() - (10 * 24 * 60 * 60)  # 10 days old
        os.utime(temp_file, (old_time, old_time))

if __name__ == "__main__":
    import time
    
    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="temp_clean_test_"))
    print(f"Created test directory: {test_dir}")
    
    try:
        # Create some sample temporary files
        create_sample_temp_files(test_dir, 8)
        
        # Clean files older than 7 days
        clean_temp_files(test_dir, days_old=7)
        
    finally:
        # Clean up test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\nRemoved test directory: {test_dir}")