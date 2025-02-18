#!/usr/bin/env python3
"""
Main module for Codebase Q&A with RAG.

This script converts a repository of code files into text,
splits the documents into manageable chunks,
embeds them using a HuggingFace model,
indexes them with Qdrant,
and sets up a RetrievalQA chain with a CodeLlama model.
An interactive Gradio interface is provided to query the codebase.
"""

import os
import argparse
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
import gradio as gr


def convert_files_to_txt(src_dir: str, dst_dir: str) -> None:
    """
    Convert files in the source directory to text files in the destination directory,
    preserving the directory structure.
    
    Args:
        src_dir (str): Source directory containing files.
        dst_dir (str): Destination directory for .txt files.
    """
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
    """
    Convert code files to text files, load the documents, and split them into chunks.

    Args:
        src_dir (str): Directory with original files.
        dst_dir (str): Directory to save text files.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: A list of document chunks.
    """
    convert_files_to_txt(src_dir, dst_dir)
    loader = DirectoryLoader(dst_dir, glob='**/*.txt', loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def setup_vectorstore(docs: list, embedding_model: str) -> Qdrant:
    """
    Initialize the embedding model and set up an in-memory Qdrant vector store.

    Args:
        docs (list): Document chunks.
        embedding_model (str): Model name for the HuggingFaceBgeEmbeddings.

    Returns:
        Qdrant: An instance of the Qdrant vector store.
    """
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model)
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",  # Use in-memory Qdrant instance
        collection_name="code_documents"
    )
    return qdrant


def setup_llm(model_name: str, max_length: int, temperature: float) -> HuggingFacePipeline:
    """
    Load a language model and create a text generation pipeline.

    Args:
        model_name (str): The model name for the tokenizer and model.
        max_length (int): Maximum length for generated text.
        temperature (float): Temperature for text generation.

    Returns:
        HuggingFacePipeline: The language model pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def setup_qa_chain(llm: HuggingFacePipeline, qdrant: Qdrant) -> RetrievalQA:
    """
    Set up a RetrievalQA chain using the language model and vector store retriever.

    Args:
        llm (HuggingFacePipeline): The language model pipeline.
        qdrant (Qdrant): The Qdrant vector store.

    Returns:
        RetrievalQA: The configured QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever()
    )
    return qa_chain


def launch_gradio_interface(qa_chain: RetrievalQA) -> None:
    """
    Create and launch a Gradio interface for querying the QA chain.

    Args:
        qa_chain (RetrievalQA): The QA chain to use for answering questions.
    """
    def answer_question(question: str) -> str:
        return qa_chain.run(question)

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.inputs.Textbox(lines=2, label="Enter your question about the codebase"),
        outputs="text",
        title="Codebase Q&A with RAG"
    )
    interface.launch()


def main():
    parser = argparse.ArgumentParser(
        description="Run the Codebase Q&A with RAG pipeline."
    )
    parser.add_argument(
        "--src-dir", type=str, default="codebase",
        help="Source directory containing the original files."
    )
    parser.add_argument(
        "--dst-dir", type=str, default="codebase_txt",
        help="Destination directory for the converted text files."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Chunk size for splitting documents."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Overlap size for splitting documents."
    )
    parser.add_argument(
        "--embed-model", type=str, default="BAAI/bge-small-en",
        help="Model name for embeddings."
    )
    parser.add_argument(
        "--llm-model", type=str, default="codellama/CodeLlama-7b-Instruct-hf",
        help="Model name for the language model."
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max length for text generation."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature for text generation."
    )

    args = parser.parse_args()

    # Setup documents, vector store, and LLM dynamically
    docs = setup_documents(args.src_dir, args.dst_dir, args.chunk_size, args.chunk_overlap)
    qdrant = setup_vectorstore(docs, args.embed_model)
    llm = setup_llm(args.llm_model, args.max_length, args.temperature)
    qa_chain = setup_qa_chain(llm, qdrant)
    launch_gradio_interface(qa_chain)


if __name__ == "__main__":
    main()
