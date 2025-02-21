import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def convert_files_to_txt(src_dir: str, dst_dir: str) -> None:
    """Convert files in the source directory to text, preserving structure."""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                new_root = os.path.join(dst_dir, os.path.dirname(rel_path))
                os.makedirs(new_root, exist_ok=True)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")
                        continue
                new_file_path = os.path.join(new_root, os.path.basename(file) + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(data)

def setup_documents(src_dir: str, dst_dir: str, chunk_size: int, chunk_overlap: int) -> list:
    """Convert files, load, and split into chunks."""
    convert_files_to_txt(src_dir, dst_dir)
    loader = DirectoryLoader(dst_dir, glob='**/*.txt', loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
