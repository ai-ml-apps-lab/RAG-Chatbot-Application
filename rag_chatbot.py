from llama_index.core import SimpleDirectoryReader
from modules.data_processing.vector_store_manager import VectorStoreManager
from modules.llm_manager import LLMManager
from modules.rag_session import RAGSession, SessionManager
import config
import sys
import logging
import gradio as gr

llm_manager = LLMManager()
session_manager = SessionManager()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

active_sessions = {}


# PROCESS DOCUMENT
def process_input(text, file, selected_model, temperature):
    """
    Processes pasted text or uploaded file.
    Builds the vector index, creates a new RAG session,
    generates a summary, and resets chat history.
    """

    try:
        if not text and not file:
            return "Please paste text or upload a File.", None, None

        if text and file:
            return "Please select only one option (Text or File).", None, None

        if text:
            text = text

        else:

            reader = SimpleDirectoryReader(input_files=[file.name])
            docs = reader.load_data()
            text = "\n".join([d.text for d in docs])

        # Setup LLM
        llm_manager.set_llm(selected_model, temperature)

        # Build vector index
        embedding = llm_manager.get_embedding()
        vector_manager = VectorStoreManager(embedding)
        index = vector_manager.build_index_from_text(text)

        # Create session
        session = RAGSession(index)
        session_id, session = session_manager.create_session(session)

        # Generate summary
        summary = session.summarize()

        chat_history = []

        return summary, session_id, chat_history

    except Exception as e:
        return f"Error: {str(e)}", None, None


# CHAT
def chat_with_doc(session_id, user_query, chat_history, model_name, temperature):
    """
    Handles user queries for an existing session.
    Retrieves relevant document context, generates a response,
    and updates the chat history.
    """
    if chat_history is None:
        chat_history = []

    if not session_id:
        chat_history.append(
            {"role": "assistant", "content": "Please process a document first."}
        )
        return chat_history

    session = session_manager.get_session(session_id)

    if not session:
        chat_history.append(
            {"role": "assistant", "content": "Session expired."}
        )
        return chat_history

    if not user_query.strip():
        return chat_history

    llm_manager.set_llm(model_name, temperature)

    response = session.chat(user_query)

    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history


# UI
def create_gradio_interface():
    """
    Creates and configures the Gradio user interface
    for the RAG chatbot application.
    """
    with gr.Blocks(
        title="Simple RAG App",
        css="""
        .gradio-container {
            max-width: 100% !important;
            padding: 12px !important;
        }
        .block {
            padding: 4px !important;
            margin: 0px !important;
        }
        footer {display:none !important;}
        """
    ) as demo:

        gr.Markdown("### RAG Chatbot Application")

        # Main Layout
        with gr.Row(equal_height=False):

            # Left placeholder
            with gr.Column():

                text = gr.Textbox(
                    label="Paste Text",
                    lines=5
                )

                input = gr.File(
                    label="Upload a File",
                    file_types=[".pdf", ".txt", ".docx", ".csv", ".md"],
                    file_count="single",
                    height=180
                )

                model_dropdown = gr.Dropdown(
                    choices=config.AVAILABLE_LLM_MODELS,
                    value=config.LLM_MODEL_ID,
                    label="Model"
                )

                temperature_slider = gr.Slider(
                    0.0, 1.0,
                    value=config.TEMPERATURE,
                    step=0.1,
                    label="Temperature",
                    info="Lower = precise, Higher = creative"
                )

                process_btn = gr.Button("Process")


            # Right placeholder
            with gr.Column():

                # Upper Right
                with gr.Row():

                    summary_box = gr.Textbox(
                        label="Summary",
                        lines=7
                    )
                    
                # Middle Right
                with gr.Row():

                    chat_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about the document...",
                        lines=3
                    )
                    
                with gr.Row():

                    send_btn = gr.Button("Send")

                # Lower Right
                with gr.Row():

                    chatbot = gr.Chatbot(label="Chat", height=200)

        # session_id = gr.Textbox(visible=False)
        session_id = gr.State()

        # EVENTS
        process_btn.click(
            fn=process_input,
            inputs=[text, input, model_dropdown, temperature_slider],
            outputs=[summary_box, session_id, chatbot],
        )

        send_btn.click(
            fn=chat_with_doc,
            inputs=[session_id, chat_input, chatbot, model_dropdown, temperature_slider],
            outputs=[chatbot],
        )

        chat_input.submit(
            fn=chat_with_doc,
            inputs=[session_id, chat_input, chatbot, model_dropdown, temperature_slider],
            outputs=[chatbot],
        )

    return demo
