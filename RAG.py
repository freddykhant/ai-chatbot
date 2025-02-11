import os

def rag_pipeline(uploaded_files: list=None):
  if uploaded_files is not None:
    for uploaded_file in uploaded_files:
      save_dir = os.path.join("uploads")
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())