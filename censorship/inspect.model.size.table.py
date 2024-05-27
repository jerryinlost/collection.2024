import os
from tabulate import tabulate

def inspect_model_sizes(root_dir):
    """
    Walk through the directory and inspect the size of each model file with specified extensions.
    Print the structured path, file name, and size in a table format.
    
    :param root_dir: The root directory containing the models.
    """
    # Define the model file extensions to look for
    model_extensions = {'.pt', '.pth', '.bin', '.onnx'}
    
    # List to store the table rows
    table_data = []
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension in model_extensions:
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert size to MB
                relative_path = os.path.relpath(dirpath, root_dir)
                
                # Exclude the root directory name from the structured path
                structured_path = relative_path.replace(f"{os.path.basename(root_dir)}/", "")
                
                # Append the row to the table data
                table_data.append([structured_path, filename, f"{file_size:.2f} MB"])

    # Print the table
    table_headers = ["Model Name (Structured Path)", "File Name", "Size"]
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))

# Example usage
root_directory = './models'
inspect_model_sizes(root_directory)