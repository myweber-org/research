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
            if file_path.suffix.lower() in extensions or file_age < cutoff_time:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_size += file_size
                except (PermissionError, OSError) as e:
                    print(f"Failed to remove {file_path}: {e}")
    
    return {
        'removed_count': len(removed_files),
        'removed_size': removed_size,
        'removed_files': removed_files
    }

def create_temp_test_files(test_dir, num_files=5):
    """Helper function to create test temporary files."""
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    for i in range(num_files):
        temp_file = test_dir / f"test_temp_{i}.tmp"
        temp_file.write_text(f"Temporary content {i}")
    
    log_file = test_dir / "app.log"
    log_file.write_text("Log entries here")
    
    return test_dir

if __name__ == "__main__":
    import time
    
    # Create a test directory with temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = create_temp_test_files(tmpdir)
        print(f"Created test files in: {test_dir}")
        
        # List files before cleanup
        print("\nFiles before cleanup:")
        for f in test_dir.iterdir():
            print(f"  {f.name}")
        
        # Perform cleanup
        result = clean_temp_files(test_dir)
        
        print(f"\nCleanup results:")
        print(f"  Removed {result['removed_count']} files")
        print(f"  Freed {result['removed_size']} bytes")
        
        # List remaining files
        print("\nFiles after cleanup:")
        for f in test_dir.iterdir():
            print(f"  {f.name}")