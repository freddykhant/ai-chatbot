import os

# method to format docs 
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def save_uploaded_files(uploaded_file: bytes, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())