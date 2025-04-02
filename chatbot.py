import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(
    page_title="Greenerway Assistant",
    layout="wide"
)

# Custom CSS for background, font, and styling
st.markdown("""
    <style>
        body {
            background-color: #112102 !important;
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background-color: #112102;
        }
        .stChatMessage {
            border-radius: 8px;
            padding: 10px;
        }
        .stChatMessage.user {
            background-color: #BCFA8E !important;
            color: black !important;
        }
        .stChatMessage.assistant {
            background-color: #1B3A17 !important;
            color: #FFFFFF !important;
        }
        .stButton>button {
            background-color: #BCFA8E !important;
            color: black !important;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    image_path = os.path.join(os.path.dirname(__file__), "logo-signal.9e6e0a8c.webp")
    st.image(image_path, width=100)
    st.title("Greenerway Assistant!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Hi :) , I am Greenerway Assistant, ask me about Greenerway")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Redirect logic for specific user queries
        if "bess size calculator" in prompt.lower():
            st.markdown("[Go to BESS Calculator](https://www.greenerway.no/calculator)")
            return
        elif "schedule of day" in prompt.lower():
            st.markdown("[Go to Scheduler App](https://scheduler-savings.streamlit.app/)")
            return
        elif "ems" in prompt.lower():
            st.markdown("[Go to EMS Dashboard](https://ems.greenerway.services/dashboard/62727b89-347c-40b0-b746-0f06ec913894)")
            return
        elif "spot prices" in prompt.lower():
            st.markdown("[Check Spot Prices](https://www.hvakosterstrommen.no/)")
            return

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
            Don't provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            result_to_show = result  # + "\nSource Docs:\n" + str(response["source_documents"])
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
