import gradio as gr
from langchain.chains import RetrievalQA
import markdown  # Import markdown for formatting

def launch_gradio_interface(qa_chain: RetrievalQA) -> None:
    """Create and launch a Gradio interface for querying the QA chain."""

    def answer_question(question: str) -> str:
        response = qa_chain.invoke({"query": question})

        # Convert response to Markdown format
        if isinstance(response, dict) and "result" in response:
            formatted_output = markdown.markdown(response["result"])  # Convert to HTML for rendering
            return formatted_output

        return response

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(lines=2, label="Enter your question about the codebase"),
        outputs=gr.Markdown(),  # Render output in Markdown format
        title="Codebase Q&A with RAG"
    )
    interface.launch(share=True)
