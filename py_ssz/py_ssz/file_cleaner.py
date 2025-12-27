
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    Files can be filtered by extension and age.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache']
    
    target_dir = Path(directory_path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    removed_files = []
    removed_size = 0
    
    for file_path in target_dir.rglob('*'):
        if file_path.is_file():
            file_age = file_path.stat().st_mtime
            should_remove = False
            
            if extensions:
                if file_path.suffix.lower() in extensions:
                    should_remove = True
            else:
                should_remove = True
            
            if should_remove and file_age < cutoff_time:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_size += file_size
                except (OSError, PermissionError) as e:
                    print(f"Failed to remove {file_path}: {e}")
    
    return {
        'removed_count': len(removed_files),
        'removed_size': removed_size,
        'removed_files': removed_files
    }

def create_temp_test_files(test_dir, num_files=5):
    """Create temporary test files for demonstration."""
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    for i in range(num_files):
        temp_file = test_dir / f"test_{i}.tmp"
        temp_file.write_text(f"Temporary content {i}")
        
        log_file = test_dir / f"app_{i}.log"
        log_file.write_text(f"Log entry {i}")
    
    return str(test_dir)

if __name__ == "__main__":
    import time
    
    test_directory = tempfile.mkdtemp()
    print(f"Created test directory: {test_directory}")
    
    create_temp_test_files(test_directory)
    
    time.sleep(1)
    
    result = clean_temp_files(test_directory, days_old=0)
    
    print(f"Cleaned {result['removed_count']} files")
    print(f"Freed {result['removed_size']} bytes")
    
    shutil.rmtree(test_directory)