import gradio as gr
from langchain.chains import RetrievalQA

def launch_gradio_interface(qa_chain: RetrievalQA) -> None:
    """Create and launch a Gradio interface for querying the QA chain."""
    def answer_question(question: str) -> str:
        return qa_chain.invoke({"question": question})

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(lines=2, label="Enter your question about the codebase"),
        outputs="text",
        title="Codebase Q&A with RAG"
    )
    interface.launch(share=True)
