import argparse
from data_processing import setup_documents
from vector_store import setup_vectorstore
from llm_pipeline import setup_llm, setup_qa_chain
from interface import launch_gradio_interface

def main():
    parser = argparse.ArgumentParser(description="Run the Codebase Q&A with RAG pipeline.")
    parser.add_argument("--src-dir", type=str, default="codebase", help="Source directory containing the original files.")
    parser.add_argument("--dst-dir", type=str, default="codebase_txt", help="Destination directory for the converted text files.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting documents.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap size for splitting documents.")
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-small-en", help="Model name for embeddings.")
    parser.add_argument("--llm-model", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Model name for the language model.")
    parser.add_argument("--max-length", type=int, default=512, help="Max length for text generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation.")
    
    args = parser.parse_args()
    
    # Setup components
    docs = setup_documents(args.src_dir, args.dst_dir, args.chunk_size, args.chunk_overlap)
    qdrant = setup_vectorstore(docs, args.embed_model)
    llm = setup_llm(args.llm_model, args.max_length, args.temperature)
    qa_chain = setup_qa_chain(llm, qdrant)

    # Launch the Gradio interface
    launch_gradio_interface(qa_chain)

if __name__ == "__main__":
    main()
