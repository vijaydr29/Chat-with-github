#this is like vectorDB where it consist of vector embedded documents

from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv() #load env will load the .env file


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') # we are extracting the openai API key and stored in OPENAI_API_KEY variable 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # and we are setting  openai api key as environment variable (environ["OPENAI_API_KEY"] )


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()


#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()