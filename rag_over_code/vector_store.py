from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Qdrant

def setup_vectorstore(docs: list, embedding_model: str) -> Qdrant:
    """Initialize the embedding model and set up a Qdrant vector store."""
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)  # Updated class

    if not docs:
        raise ValueError("No documents found for vector store indexing!")

    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",  # Use in-memory Qdrant instance
        collection_name="code_documents"
    )
    return qdrant
