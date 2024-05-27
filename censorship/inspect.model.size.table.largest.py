import os
from tabulate import tabulate

def get_largest_model_per_directory(root_dir):
    """
    Walk through the directory and find the largest model file in each subdirectory.
    Print the structured path and file size in a table format.
    
    :param root_dir: The root directory containing the models.
    """
    # Define the model file extensions to look for
    model_extensions = {'.pt', '.pth', '.bin', '.onnx'}
    
    # Dictionary to store the largest model file in each directory
    largest_models = {}

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        largest_file = None
        largest_file_size = 0
        
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension in model_extensions:
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path)  # Size in bytes
                
                # Check if this is the largest file found in this directory
                if file_size > largest_file_size:
                    largest_file = filename
                    largest_file_size = file_size

        if largest_file:
            relative_path = os.path.relpath(dirpath, root_dir)
            structured_path = relative_path.replace(f"{os.path.basename(root_dir)}/", "")
            largest_models[structured_path] = largest_file_size

    # Prepare the table data
    table_data = [[path, f"{size / (1024 * 1024):.2f} MB"] for path, size in largest_models.items()]

    # Print the table
    table_headers = ["Model Name (Structured Path)", "Size"]
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

# Example usage
root_directory = './models'
get_largest_model_per_directory(root_directory)