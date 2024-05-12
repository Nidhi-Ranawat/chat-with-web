import pinecone
import sys
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import pytube as pt
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone

import json

# url = "https://www.youtube.com/watch?v=sEirPozuvX4"
import whisper
load_dotenv()
pinecone_apikey = os.environ.get("PINECONE_API_KEY")
replicate_apikey = os.environ.get("REPLICATE_API_TOKEN")

# pinecone_apikey = os.environ.get("PINECONE_API_KEY")

pinecone = Pinecone(
    api_key=pinecone_apikey,environment="gcp-starter"
)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# def generate_unique_filename(base_name, extension):
#     index = 1
#     while True:
#         filename = f"{base_name}{index}.{extension}"
#         if not os.path.exists(filename):
#             return filename
#         index += 1

def generate_unique_filename(base_name, extension):
    index = 1
    while True:
        filename = f"{base_name}{index}.{extension}"
        if not os.path.exists(filename):
            return filename
        index += 1

def get_vectorstore_from_url(url):

    try:
        with open("data.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    if url in data or url is None:
        print("url already exists or is None...")
        vector_store = PineconeStore.from_existing_index(index_name="urls",embedding=embeddings)
        return vector_store

    # youtube urls
    if "youtube" in url.lower():
        yt = pt.YouTube(url)
        video_title = yt.title
        stream = yt.streams.filter(only_audio=True)[0]
        filename = generate_unique_filename(video_title, "mp3")
        stream.download(filename=filename)
        model = whisper.load_model("base")
        result = model.transcribe(filename)
        text = result["text"]
        # doc =  Document(page_content=text, metadata={"source": "local"})# print(doc)
        
        docs=[]
        docs.append(Document(
            page_content=text,
            metadata={"source": "local"}
        ))
    
    # web urls/invalid urls
    else:
        try:
        # get the text in document form
            loader = WebBaseLoader(url)
            docs = loader.load()
        except:
            st.write("Invalid URL")
            vector_store = PineconeStore.from_existing_index(index_name="urls",embedding=embeddings)
            return vector_store

    # if url is valid store and retrive Vectors flag = True
    data.append(url)

    with open("data.json", "w") as json_file:
        json.dump(data, json_file)

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    
    # create a vectorstore from the chunks
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name="urls")
       
    # index = pinecone.Index("urls")
    return vector_store

def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()  
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )
    return qa_chain

def get_response(user_input):
    print("getting response")
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    response = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "question": user_query
    })

    print(response["answer"])
    return response['answer']

# app config
st.set_page_config(page_title="Chat with URLs", page_icon="🤖")

# Define section titles
section_titles = ['Submit URLs', 'Q & A']
selected_section = st.sidebar.radio("Select Section", section_titles)

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I can help you to query URLs"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(None)  

# Render selected section
if selected_section == 'Submit URLs':
    st.header('Submit the website links or youtube links')
    website_url = st.text_input("Website URL")
    button = st.button("Submit")

    # Convert button
    if button:
        with st.spinner('Wait for it...'):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)  

        st.success('Done!')

elif selected_section == 'Q & A':
    st.header('Query websites')
    # conversation
    user_query = st.chat_input("Ask your query here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        # sessions for code reload
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)