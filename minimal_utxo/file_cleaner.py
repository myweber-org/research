
import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    lines_removed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                if line not in seen_lines:
                    f.write(line)
                    seen_lines.add(line)
                else:
                    lines_removed += 1
        
        print(f"Successfully processed '{input_file}'")
        print(f"Original lines: {len(lines)}")
        print(f"Unique lines: {len(seen_lines)}")
        print(f"Duplicate lines removed: {lines_removed}")
        print(f"Output saved to: '{output_file}'")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        print("If output_file is not specified, input_file.deduped will be used")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = remove_duplicates(input_file, output_file)
    sys.exit(0 if success else 1)