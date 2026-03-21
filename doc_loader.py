'''Handles document reading
    -- read file
    return text'''

def load_document(file_path):
    with open(file_path, "r") as file:
        return file.read()