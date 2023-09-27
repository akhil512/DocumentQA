import os
import yaml
import base64
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import GPT4All
from langchain import PromptTemplate
from QuestionAnswer import DocumentQA
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('config_file.yml','r') as f:
    config = yaml.safe_load(f)

@st.cache_resource
def on_load(template):

    vector_store = FAISS
    llm = GPT4All(
        max_tokens = 256,
        model= config['model_path'],
        top_k=40,
        top_p=0.4,
        temp=0.4
                )

    embeddings = SentenceTransformerEmbeddings(
        model_name= config['embeddings_path']
                                            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config['chunk_size'],
        chunk_overlap  = config['chunk_overlap'],
        length_function = len,
    )

    prompt = PromptTemplate(
        input_variables=['context','question'],
        template=template
                            )

    return {
    'model':llm,
    'embeddings':embeddings,
    'prompt':prompt,
    'splitter':text_splitter,
    'store':vector_store
}

def on_update():
    for i in os.listdir(config['vector_store_path']):
        os.remove(config['vector_store_path']+'/'+i)
    st.session_state['flag'] = True

st.set_page_config (initial_sidebar_state="expanded",layout="wide")
st.header("Welcome to Document Answering System")
st.info("This is a simple web app to answer questions from a given document.Please upload your document in the sidebar and ask your question in the text box below.")

template = """You are given a task which will be described below
  Instruction:
  The task is to answer the questions asked by me in the given context below:
  {context}
  you should strictly answer the questions within the given context and in a paragraph format otherwise respond don't know.
  Don't generate question answers just answer the question in the given context.
  {question}
  """
args = on_load(template)

st.sidebar.title("Upload your document here")
st.sidebar.write("Please upload your document here")

uploaded_files = st.sidebar.file_uploader("Upload your file or files",type=['pdf'],accept_multiple_files=True,on_change=on_update)

if uploaded_files:

    args['directory'] = uploaded_files
    doc_qa = DocumentQA(**args)
    doc_qa.pre_processor()
    query = st.text_input("Ask your question here")

    if query:
        doc_qa.get_query(query)
        output = doc_qa.run(query)
        st.subheader("Question")
        st.write(output['question'])
        st.subheader("Answer")
        st.write(output['output_text'])
        st.subheader("Relevent text for the query")
        st.write(output['input_documents'])