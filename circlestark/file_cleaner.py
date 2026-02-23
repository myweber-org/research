import sys

def remove_duplicates(input_file, output_file):
    seen_lines = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen_lines:
                    seen_lines.add(stripped_line)
                    unique_lines.append(line)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except IOError as e:
        print(f"Error reading file: {e}")
        return False
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        print(f"Successfully removed duplicates. Output saved to '{output_file}'")
        return True
    except IOError as e:
        print(f"Error writing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    remove_duplicates(input_path, output_path)