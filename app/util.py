import os

# Save uploaded file
def save_uploaded_file(uploaded_file):
    save_path = os.path.join("uploads", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path
