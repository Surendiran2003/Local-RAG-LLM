import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

# Constants
OLLAMA_URL = "http://ollama:11434"
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "all-minilm"

# Streamlit App Title
st.sidebar.title('RAG App')

# Functions

def initialize_llm():
    """Initialize the LLM and embeddings."""
    llm = Ollama(base_url=OLLAMA_URL, model=LLM_MODEL)
    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBEDDING_MODEL)
    return llm, embeddings

def create_vector(text, embeddings):
    """Create the vector from the text."""
    return FAISS.from_documents(text, embeddings)

def generate_prompt():
    """Generate the prompt template."""
    return """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
        3. Keep the answer crisp and limited to 3-4 sentences.

        Context: {context}

        Question: {question}

        Helpful Answer:
    """

def process_uploaded_file(uploaded_file):
    """Process the uploaded PDF file and return the extracted documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    os.unlink(temp_path)
    return docs

def setup_qa_chain(llm, retriever):
    """Set up the QA chain."""
    prompt_template = PromptTemplate.from_template(generate_prompt())

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )

    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
    )

# App Workflow
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Initialize LLM and embeddings
    llm, embeddings = initialize_llm()

    # Process the uploaded file
    with st.spinner('Processing the PDF...'):
        docs = process_uploaded_file(uploaded_file)

    # Create a vector from the text content
    with st.spinner('Creating the vector...'):
        vector = create_vector(docs, embeddings)
        st.sidebar.success("Vector created successfully")

    # User Query
    query = st.text_input("Enter the query")

    if st.button('Submit Query', type='primary'):
        # Define retriever and QA chain
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        qa_chain = setup_qa_chain(llm, retriever)

        # Generate and display the answer
        with st.spinner('Generating the answer...'):
            answer = qa_chain.invoke(query)
        
        st.write(answer["result"])
else:
    st.sidebar.info("Please upload a PDF file to proceed.")
