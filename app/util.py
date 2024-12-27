import os

def save_uploaded_file(file, save_dir="uploads/"):
    """Save uploaded file."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path
