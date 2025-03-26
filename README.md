# Greenerway Chatbot
This project provides a question answering bot that can process PDF documents and answer questions based on their content. 
It utilizes Langchain, Hugging Face models(mistralai/Mistral-7B-Instruct-v0.3), and a FAISS vector database. 
A web interface for interacting with the assistant is built using Streamlit.

## Project Structure
The project consists of the following files:
- `create_knowledge-base.py`: This script loads PDF files from the `data/` directory, splits them into chunks, creates embeddings using a Hugging Face model, and stores these embeddings in a FAISS database.
- `use_knowledge-base.py`: This script loads the FAISS database and a Hugging Face language model to create a question answering chain. It takes a user query from the command line and provides an answer along with the source documents.
- `chatbot.py`: This script builds a web interface using Streamlit that allows users to interact with the question answering system.
- `requirements.txt`: This file lists the Python packages and their versions required to run the project.
- `data/`: ) This directory contains the PDF files you want to process.
- `vectorstore/`: This directory contains the FAISS database.
- `.env`:  This file can be used to store environment variables, I have stored my Hugging Face API token. Remember not to rexpose your API Token.

## Setup

1.  **Clone the repository** 
## Usage (With Docker)
1.  **Ensure Docker is installed and running on your system.**
2.  **Build the Docker image:**
    Navigate to the root of your project directory in your terminal and run:
    ```bash
    docker build -t greenerway_chatbot .
    ```
3.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 greenerway_chatbot
    ```
    This command will:
    * Run a container from the `greenerway_chatbot` image.
    * Map port `8501` of the container to port `8501` on your host machine, allowing you to access the Streamlit application in your browser.
    * It assumes your `HF_TOKEN` is in the `.env` file, which is copied into the container during the build process.
    ```
    
4.  **Access the application:**
    Open your web browser and go to `http://localhost:8501`.
