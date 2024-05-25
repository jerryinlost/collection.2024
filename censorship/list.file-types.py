import os
from collections import defaultdict

def list_file_types(root_folder):
    file_types = defaultdict(int)
    
    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # Extract file extension
            file_extension = os.path.splitext(filename)[1].lower()
            file_types[file_extension] += 1
    
    return file_types

def print_file_types(file_types):
    print("File types and their counts:")
    for file_type, count in file_types.items():
        print(f"{file_type}: {count}")

if __name__ == "__main__":
    # Specify the root folder to start walking
    root_folder = "path/to/your/folder"
    
    # Get the file types
    file_types = list_file_types(root_folder)
    
    # Print the file types and their counts
    print_file_types(file_types)