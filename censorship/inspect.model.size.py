import os

def inspect_model_sizes(root_dir):
    """
    Walk through the directory and inspect the size of each model file with specified extensions.
    Print the structured path if models with specified extensions are found.
    
    :param root_dir: The root directory containing the models.
    """
    # Define the model file extensions to look for
    model_extensions = {'.pt', '.pth', '.bin', '.onnx'}
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        found_models = False
        
        # Check each file in the directory
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension in model_extensions:
                found_models = True
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert size to MB
                print(f"Found model: {file_path} (Size: {file_size:.2f} MB)")
        
        if found_models:
            # Print the structured path relative to the root directory
            relative_path = os.path.relpath(dirpath, root_dir)
            print(f"Structured path: {relative_path}\n")

# Example usage
root_directory = './models'
inspect_model_sizes(root_directory)