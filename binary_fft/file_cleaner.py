import sys
import os

def clean_file(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    if output_path is None:
        output_path = input_path + ".cleaned"
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        cleaned_lines = []
        seen = set()
        
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                cleaned_lines.append(stripped + '\n')
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(cleaned_lines)
        
        print(f"Successfully cleaned file. Output saved to: {output_path}")
        print(f"Original lines: {len(lines)} -> Cleaned lines: {len(cleaned_lines)}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_file(input_file, output_file)