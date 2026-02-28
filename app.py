from rag_chatbot import create_gradio_interface

if __name__ == "__main__":
    """
    Launches the Gradio-based RAG chatbot interface.
    """
    demo = create_gradio_interface()
    demo.launch(inbrowser=True)