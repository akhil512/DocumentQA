import os
import yaml
import pickle
import time
from PyPDF2 import PdfReader
from langchain.llms import GPT4All
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('config_file.yml','r') as f:
    config = yaml.safe_load(f)

class DocumentQA:

    def __init__(self,**kwargs):
        self.model = kwargs.get('model')
        self.embeddings = kwargs.get('embeddings')
        self.prompt = kwargs.get('prompt')
        self.directory = kwargs.get('directory')
        self.splitter = kwargs.get('splitter')
        self.store = kwargs.get('store')
        self.chain = load_qa_chain(llm=self.model,chain_type='stuff',prompt=self.prompt,verbose=False)
        self.vector_store = None

    def read_pdf(self):
        self.text = ''
        files= self.directory
        if len(files) == 1:
            loader = PdfReader(files[0])
            for page in loader.pages:
                self.text += page.extract_text()
        elif len(files)>1:
            for i,file in enumerate(files):
                loader = PdfReader(file)
                for page in loader.pages:
                    self.text += page.extract_text()

    def text_splitter(self):
        self.chunked_text = self.splitter.split_text(self.text)

    def vectorstore(self):
        self.vector_store = self.store.from_texts(self.chunked_text,self.embeddings)

    def get_query(self,que):
        self.query = que

    def search(self):
        self.relevent_text = self.vector_store.similarity_search(self.query,k=1)

    def get_answer(self):
        return self.chain({"input_documents":self.relevent_text,"question":self.query})

    def pre_processor(self):
        if len(os.listdir(config['vector_store_path'])) == 0:
            print('Reading pdf started....')
            self.read_pdf()
            print('Reading pdf completed!')
            print('Splitting text started....')
            self.text_splitter()
            print('Splitting text completed!')
            print('Vectorizing text started....')
            self.vectorstore()
            print('Vectorizing text completed!')
            # os.system('mkdir '+config['vector_store_path'])
            with open(config['vector_store_path']+'/'+self.directory[0].name.split('.')[0]+'.pkl','wb') as f:
                pickle.dump(self.vector_store,f)
        else:
            if os.path.exists(config['vector_store_path']+'/'+self.directory[0].name.split('.')[0]+'.pkl'):
                with open(config['vector_store_path']+'/'+self.directory[0].name.split('.')[0]+'.pkl','rb') as f:
                    print('Loading vector store started....')
                    self.vector_store = pickle.load(f)
                    print('Loading vector store completed!')

    def run(self,query):
        self.get_query(que=query)
        self.search()
        start_time = time.time()
        answer = self.get_answer()
        end_time = time.time()
        print('Time taken to answer the question: ',end_time-start_time)
        return answer

