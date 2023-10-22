import streamlit as st
import pandas as pd
import openai
import os
import PyPDF2
import tiktoken
import urllib.request
import requests
from io import BytesIO
from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = 'sk' +  '-CGl'+ 'BQwoWejfc5lMFchXgT3BlbkFJKX2jpJIbqzQGBvupeA7p'
pdfURL="https://www.alida.com/resources/the-product-leaders-guide-thanks?submissionGuid=a40d4795-3e6d-4dc8-be28-6685331f94f8"
pdfURL="https://arxiv.org/pdf/1706.03762.pdf"

def fetch_remote_pdf(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return BytesIO(response.content)

def extract_text_from_pdf(buffer):
    reader = PdfReader(buffer)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_buffer = fetch_remote_pdf(pdfURL)
pdf_text = extract_text_from_pdf(pdf_buffer)
#print(pdf_text)



text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100,length_function=len, is_separator_regex=False)
docs = text_splitter.create_documents([pdf_text])
print(docs)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)

def similarity_search(query = "How do we build a successful community?"):
  docs = db.similarity_search(query)
  return docs

def retrieval_augmented_generation(query = "How do we build a successful community?"):
  docs = db.similarity_search(query)
  chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
  query = "WHow do we build a successful community?"
  return chain.run(input_documents=docs, question=query)

dlist=similarity_search("How do we build a successful community?")
for d in dlist:
  print(d)


st.write("# It works!!")