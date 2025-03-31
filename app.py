from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from data.employees import generate_employee_data
from gui import AssistantGUI
from assistant import Assistant
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
import logging
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


if __name__ == "__main__":

    load_dotenv()

    st.set_page_config(page_title="SaranAILabs", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")
    
    @st.cache_data(ttl=600, show_spinner="Generting User Data...")
    def get_user_data():
        users = generate_employee_data(1)[0]
        return users
    
    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)

            embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

            vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=embedding_function,
            )
            return vectorstore
        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vector store: {str(e)}")
            return None

    vector_store = init_vector_store("./data/SaranAILabs2.pdf")
    if vector_store is None:
        st.error(
            "Failed to initialize vector store. Please check the logs for more information."
        )
        st.stop()   

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    system_prompt = SYSTEM_PROMPT
    welcome_message = WELCOME_MESSAGE
    customer_data = get_user_data()
    #vector_store = init_vector_store("./data/SaranAILabs2.pdf")

    if "customer" not in st.session_state:
        st.session_state.customer = customer_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": welcome_message}]

    assistant = Assistant(
        system_prompt=system_prompt,
        llm=llm,
        employee_information=st.session_state.customer,
        message_history=st.session_state.messages,
        vector_store=vector_store,
    )

    gui = AssistantGUI(assistant=assistant)
    gui.render()