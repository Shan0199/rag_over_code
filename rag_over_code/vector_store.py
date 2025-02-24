# vector_store.py

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant

from transformers import AutoModel, AutoTokenizer
import torch

def setup_vectorstore(documents, embedding_model_name):
    # Initialize the tokenizer and model with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True)

    # Function to compute embeddings
    def embed_function(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Access the first element of the tuple
        embeddings = outputs[0]
        return embeddings

    # Initialize HuggingFaceEmbeddings with the custom embed_function
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        embed_function=embed_function
    )

    # Create Qdrant vector store from documents
    qdrant = Qdrant.from_documents(documents, embeddings)
    return qdrant
