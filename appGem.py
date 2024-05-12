import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

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
google_apikey = os.environ.get("GOOGLE_API_KEY")

pinecone = Pinecone(
    api_key=pinecone_apikey,environment="gcp-starter"
)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

llm = genai.GenerativeModel('gemini-pro')

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
        video_title = video_title.replace(":","-")
        print("yt.title", video_title)
        audio_streams = yt.streams.filter(only_audio=True)
        if audio_streams:
            stream = audio_streams[0]
            filename = video_title + ".mp3"
            stream.download(filename=filename)
            model = whisper.load_model("base")
            result = model.transcribe(filename)
            print("filename", filename)
            text = result["text"]
        else:
            print("No audio stream available for the provided YouTube video.")
        
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

# def get_context_retriever_chain(vector_store):
#     model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_apikey,
#     temperature=0.2,convert_system_message_to_human=True)
    
#     retriever = vector_store.as_retriever()  

#     template = """
#     Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
#     {context}
#     Question: {question}
#     Helpful Answer:
#     """

#     prompt = PromptTemplate.from_template(template)

#     # If there is no chat_history, then the input is just passed directly to the retriever. 
#     qa_chain = RetrievalQA.from_chain_type(
#         model,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )
#     return qa_chain
  
def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",    convert_system_message_to_human=True)
    
    retriever = vector_store.as_retriever()  

    PROMPT_TEMPLATE = """
    Current conversation: {chat_history}
    Human: {input}
    Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.

    AI Assistant:
    """

    PROMPT = PromptTemplate(
        input_variables=["chat_history", "input"], template=PROMPT_TEMPLATE
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, PROMPT)
    return retriever_chain  

def get_conversational_rag_chain(retriever_chain): 
    llm = ChatGoogleGenerativeAI(model="gemini-pro",    convert_system_message_to_human=True)
    
    PROMPT_TEMPLATE = """T"Answer the user's questions based on the below context:\n\n{context}"

    Current conversation:
    {chat_history}
    Human: {input}
    AI Assistant: 
    """

    PROMPT = PromptTemplate(
        input_variables=["context","chat_history", "input"], template=PROMPT_TEMPLATE
    )

    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm,PROMPT)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_relevant_docs(user_input):
#     print("getting response")
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

#     retrieved_docs = retriever_chain.invoke({
#         "input": user_input,
#         "chat_history": st.session_state.chat_history
#     })
#     # return retrieved_docs

def get_response(user_input):
    print("getting response")
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })
    return response["answer"]

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