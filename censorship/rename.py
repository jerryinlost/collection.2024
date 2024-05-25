import os

folder_path = os.path.abspath('Docker')
print(folder_path)
for filename in os.listdir(folder_path):
    if filename.endswith('.rar') and filename.startswith("1."):
        parts = filename.split('.')
        prefix = parts[0].zfill(2)
        cnt = 15
        if len(parts) == 4:
            new_filename = f"{prefix}.{parts[1]}-{cnt}.{parts[-2]}.{parts[-1]}"
        elif len(parts) == 5:
            new_filename = f"{prefix}.{parts[1]}.{parts[2]}-{cnt}.{parts[-2]}.{parts[-1]}"  
        print(new_filename)
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))