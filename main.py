from flask import Flask, Blueprint, jsonify, request
from rich import print, pretty
import datetime
import requests
import os
import json
import re
import langchain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain import PromptTemplate, LLMChain
from typing import List
from io import BytesIO
import pandas as pd
import pinecone
import openai
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import hashlib
from langchain.chat_models import ChatOpenAI
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
PINECONE_INDEX_NAME = 'chatbot'
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
openai.openai_api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

def parse_srt(file): 
    lines = file.read().decode("utf-8")
    text = []
    text.append(lines)
    return text

def parse_csv(file):
    data = file.read()
    string_data = str(data)
    text = []
    text.append(string_data)
    return text

def correct_grammar(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Correct each sentence separately
    corrected_sentences = []
    for sentence in sentences:
        # Tokenize the sentence into words and tag their parts of speech
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)

        # Perform grammar correction based on POS tags
        corrected_words = []
        for word, tag in tagged_words:
            # Perform grammar correction as needed
            # Example correction: singularize nouns, use proper verb forms, etc.
            corrected_word = word  # Placeholder for correction logic
            corrected_words.append(corrected_word)

        # Reconstruct the corrected sentence
        corrected_sentence = " ".join(corrected_words)
        corrected_sentences.append(corrected_sentence)

    # Reconstruct the entire corrected text
    corrected_text = " ".join(corrected_sentences)
    return corrected_text

def text_to_docs(text:str, filename:str):
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for i, doc in enumerate(page_docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=50,
        )
        if doc.page_content == "":
            continue
        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={
                    "page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{filename}"
            doc_chunks.append(doc)
    return doc_chunks

def create_vector(docs):
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)

def delete_vectore(source):
    index = pinecone.Index(PINECONE_INDEX_NAME)
    return index.delete(
        filter={
            "source": f"{source}",
        }
    )

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(
            manager="map", data_dir=f"map/map_cache_{hashed_llm}"),
    )

def generate_message(query, history, behavior, temp):
    template = """ {behavior}
    
    Training data: {examples}

    Chathistory: {history}
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "examples", "human_input", "behavior"], template=template)

    langchain.llm_cache = GPTCache(init_gptcache)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                    temperature=temp,
                    openai_api_key=OPENAI_API_KEY)

    conversation = LLMChain(
        llm=llm,
        verbose=True,
        prompt=prompt
    )
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    _query = query
    docs = docsearch.similarity_search(query=_query, k=10)

    examples = ""
    for doc in docs:
        doc.page_content = doc.page_content.replace('\n\n', ' ')
        examples += doc.page_content + '\n'

    response = conversation.run(
        human_input=query,
        history=history,
        behavior=behavior,
        examples=examples
    )

    return response


@app.route("/")
def hello_world():
    return "Here!"

@app.route('/sendfile', methods=['POST'])
def create_train_file():
    try:
        file = request.files.get('file', None)
        if not file:
            return {"success": False, "message": "Invalid file"}
        filename = secure_filename(file.filename)
        file.save(filename)
        with open(filename, 'rb') as f:
            if (filename.split('.')[-1] == 'json'):
                output = parse_srt(f)
            elif (filename.split('.')[-1] == 'csv'):
                output = parse_csv(f)
            
            result = text_to_docs(output, filename)
            print(result)
            create_vector(result)   #
            console.log(jsonify({
                'success': True,
                'code': 200,
                'message': 'Training is successful!',
                'data': filename
            }))
            return jsonify({
                'success': True,
                'code': 200,
                'message': 'Training is successful!',
                'data': filename
            })
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.route('/getReply', methods=['POST'])
def get_reply():
    query = request.form["message"]
    temp = json.loads(request.form["history"])
    chat_history = None
    history = [{'role': 'ai', 'content': 'Hello!'}]
    for x in temp:
        if x['type'] == 0:
            history.append({'role': 'human', 'content': x['text']})
        else:
            history.append({'role': 'ai', 'content': x['text']})
    if len(history) > 8:
        chat_history = history[-4:]
    response = generate_message(query, 
                                history, 
                                "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown.",
                                0.5)
    return response

if __name__ == '__main__':
    app.run(debug = True)

