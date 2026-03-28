import os
import zipfile

def zip_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'Backend', 'data', 'index')
    zip_path = os.path.join(base_dir, 'Backend', 'data', 'scoutsearch_data.zip')
    
    # We want to zip everything in Backend/data/index that is a .json file, including barrels
    print(f"Creating {zip_path}...")
    
    file_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    # arcname should be relative to Backend/data/index
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
                    file_count += 1
                    
    print(f"Successfully compressed {file_count} files into {zip_path}!")
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Zip file size: {size_mb:.2f} MB")
    
if __name__ == '__main__':
    zip_data()
