from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant

def setup_vectorstore(docs: list, embedding_model: str) -> Qdrant:
    """Initialize embedding model and vector store."""
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",  # Use in-memory Qdrant instance
        collection_name="code_documents"
    )
    return qdrant
