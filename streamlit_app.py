import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp"
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

import nltk

@st.cache_data
def get_stopwords():
    nltk.download('stopwords')

st.set_page_config(page_title="Chat with a friend on Quantitative Physics", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Ask Me a Question about Quantitative Physics")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Quantitative Physics!!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()

    Settings.chunk_size = 1500
    Settings.chunk_overlap = 50
    Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    api_key=st.secrets.google_gemini_key,
    )

    
    Settings.llm = Gemini(
        model="models/gemini-2.5-flash",
        temperature=1.0,
        system_prompt="""You are an expert on Quantitative Physics.
        Answer the question using the provided documents, which contain relevant textbook text.
        The context for all questions is the field of Quantitative Physics. Whenever possible, include a quotation from the provided excerpts of the work to illustrate your point.
        Respond using a educational but direct tone, typical of a high school teacher.
        Keep your answers under 500 words.""",
        api_key = st.secrets.google_gemini_key,
        safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
],
    )
 

    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=False,
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = ""
        try:
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
        except:
            st.error("We got an error from Google Gemini - this may mean the question had a risk of producing a harmful response. Consider asking the question in a different way.")        
        if response_stream != "":
            with st.spinner("waiting"):
                try:
                    st.write_stream(response_stream.response_gen)
                except:
                    st.error("We hit a bump - let's try again")
                    try:
                        resp = st.session_state.chat_engine.chat(prompt)[0]
                        st.write(resp)
                    except:
                        st.error("We got an error from Google Gemini - this may mean the question had a risk of producing a harmful response. Consider asking the question in a different way.")
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)
