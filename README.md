# RAG Chatbot Application

This project is a Retrieval-Augmented Generation (RAG) application built using the LlamaIndex framework and OpenAI models.
It allows users to:

•	Paste text or upload a document

•	Automatically generate a summary

•	Ask questions about the document

•	Chat with contextual memory

•	Switch models and adjust temperature dynamically


It is a Generative AI application implementing a modular and session-based RAG pipeline. Like other RAG applications, this is a document-based LLM system with customizable options to configure model responses.
________________________________________
## What This App Does

Instead of training a model, the app:

1.	Converts your document into embeddings
2.	Stores them in a vector index
3.	Retrieves relevant parts when you ask a question
4.	Sends the retrieved context + your question to the LLM
5.	Returns a grounded, document-based response
________________________________________
## How It Works (Step by Step)

1.	Paste Text or Upload File

•	Paste raw text

•	OR upload .pdf, .txt, .docx, .csv, .md

•	Click Process

 <img width="973" height="445" alt="image" src="https://github.com/user-attachments/assets/a9850864-4149-49cb-b38c-c696844f4c09" />

________________________________________
2.	Automatic Summary
   
After processing:

•	The document is embedded

•	A vector index is created

•	A new session is generated

•	A summary appears automatically

Previous chat history is cleared when a new document is processed.

 <img width="975" height="463" alt="image" src="https://github.com/user-attachments/assets/25bd4190-ed61-4ad0-a491-04e8f97f7afa" />

________________________________________
3.	Ask Questions & Chat with Memory
   
Type your question and click Send.

The system:

•	Retrieves relevant document chunks

•	Sends them to the LLM

•	Returns a contextual answer

The app:

•	Maintains conversation history

•	Keeps context within the same document session

•	Allows follow-up questions

Each document has its own isolated session.


 <img width="974" height="449" alt="image" src="https://github.com/user-attachments/assets/3a95d11e-06d8-4ce5-ae2a-4d5bd653dc5b" />

________________________________________
4.	Switch Model Anytime
   
You can change:

•	The LLM model

•	The temperature value

This allows:

•	More precise answers (low temperature)

•	More creative answers (high temperature)

•	Easy model comparison without restarting the app


 <img width="975" height="447" alt="image" src="https://github.com/user-attachments/assets/9cd59300-437d-482c-98f9-7980cbf950e9" />

________________________________________
## Architecture (Simple View)

Document → Embeddings → Vector Index → RAG Session → LLM Response

Main components:

•	LLM Manager

•	Vector Store Manager

•	RAG Session Handler

•	Gradio UI Layer

________________________________________
## Installation

git clone https://github.com/ai-ml-apps-lab/RAG-Chatbot-Application

cd RAG-Chatbot-Application

pip install -r requirements.txt

________________________________________
## Setup

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here

________________________________________
## Run

python app.py

The Gradio interface will open in your browser.

________________________________________
