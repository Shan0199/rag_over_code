from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
import logging
def setup_llm(model_name: str, max_length: int, temperature: float) -> HuggingFacePipeline:
    """Load a language model and create a text generation pipeline."""
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    if not device:
        logging.info("Running LLM on GPU")
    else:
        logging.warning("Running LLM on  CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_kwargs={"max_length": max_length, "temperature": temperature}
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def setup_qa_chain(llm: HuggingFacePipeline, qdrant) -> RetrievalQA:
    """Set up a RetrievalQA chain using LLM and vector store."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever()
    )
