from pathlib import Path
import streamlit as st
import uuid
from utils import load_config  
from service import DocumentProcessingService  
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from service import QueryService  
from collections import defaultdict

def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

if 'users' not in st.session_state:
    st.session_state['users'] = recursive_defaultdict()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

@st.cache_resource()
def cached_load_config():
    config = load_config()
    return config

@st.cache_resource()
def load_llama_2_llm(config):
    model_path = hf_hub_download(repo_id=config['llm_repo_id'], filename=config['llm_file_name'], resume_download=True)
    return LlamaCpp(model_path=model_path, n_gpu_layers=config['n_gpu_layers'], n_batch=config['n_batch'],
                    verbose=True, f16_kv=True, callback_manager=callback_manager, n_ctx=config['n_ctx'])

@st.cache_resource()
def load_hf_embedding(config):
    return HuggingFaceEmbeddings(model_name=config['data_ingestion']['embed_model'], model_kwargs={'device': config['device']})

def get_models(config):
    if config.get("USE_OPENAI_MODEL", True):
        llm = OpenAI()
        embeddings = OpenAIEmbeddings()
    else:
        llm = load_llama_2_llm(config)
        embeddings = load_hf_embedding(config)
    return llm, embeddings

config = cached_load_config()
llm, embeddings = get_models(config)

@st.cache_data()
def get_document_processing_service():
    document_service = DocumentProcessingService()
    return document_service

document_processing_service = get_document_processing_service()

def create_new_user():
    new_user_id = str(uuid.uuid4())
    st.session_state['users'][new_user_id] = {'conversations': recursive_defaultdict()}
    return new_user_id

def create_new_conversation(user_id):
    new_convo_id = str(uuid.uuid4())
    st.session_state['users'][user_id]['conversations'][new_convo_id] = {'messages': []}
    return new_convo_id


def manage_users_ui():
    with st.sidebar:
        st.header("Users")
        if st.button("Add New User"):
            new_user_id = create_new_user()
            st.session_state['selected_user_id'] = new_user_id
        user_ids = list(st.session_state['users'].keys())
        selected_user_id = st.selectbox("Select User", options=[""] + user_ids, index=0)
        
        if selected_user_id:
            st.session_state['selected_user_id'] = selected_user_id

def manage_conversations_ui():
    user_id = st.session_state.get('selected_user_id')
    if user_id and user_id != "Select a user...":
        with st.sidebar:
            st.header("Conversations")
            if st.button("Add New Conversation"):
                new_convo_id = create_new_conversation(user_id)
                st.session_state['selected_conversation_id'] = new_convo_id

            for convo_id in st.session_state['users'][user_id]['conversations']:
                if st.button(f"Conversation {convo_id}", key=convo_id):
                    st.session_state['selected_conversation_id'] = convo_id

def conversation_ui():
    """UI for managing individual conversations."""
    conversation_id = st.session_state.get('selected_conversation_id')
    if conversation_id:
        user_id = st.session_state['selected_user_id']
        st.header(f"Conversation: {conversation_id}")
        
        file_uploader_key = f"file-uploader-{conversation_id}"
        uploaded_files = st.file_uploader("Upload document", type=['pdf', 'docx', 'txt'], accept_multiple_files=True, key=file_uploader_key) # , key=file_uploader_key
        if uploaded_files:
            st.success("Files uploaded successfully!")

        messages = st.session_state['users'][user_id]['conversations'][conversation_id]['messages']
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("What is up?"):
            process_user_input(prompt, uploaded_files, user_id, conversation_id)

def process_user_input(prompt, uploaded_files, user_id, conversation_id):
    """Process user input and manage the conversation state."""
    with st.chat_message("user"):
        st.markdown(prompt)

    conversations = st.session_state['users'][user_id]['conversations']
    if 'messages' not in conversations[conversation_id] or not isinstance(conversations[conversation_id]['messages'], list):
        conversations[conversation_id]['messages'] = []
        
    messages = conversations[conversation_id]['messages']
    messages.append({"role": "user", "content": prompt})

    document_processing_service = get_document_processing_service()
    if uploaded_files:
        for uploaded_file in uploaded_files:
            document_processing_service.process_uploaded_file(file=uploaded_file, user_id=user_id, conversation_id=conversation_id, embeddings=embeddings)
        db_path = Path(config["vector_store_dir"], user_id, conversation_id)
    else:
        db_path = None

    if 'conversation_memory' not in st.session_state['users'][user_id]['conversations'][conversation_id]:
        memory = document_processing_service.initialize_conversation_memory(user_id, conversation_id, llm)
        st.session_state['users'][user_id]['conversations'][conversation_id]['conversation_memory'] = memory
    else:
        memory = st.session_state['users'][user_id]['conversations'][conversation_id]['conversation_memory']

    query_service = QueryService(db_path=db_path, embeddings=embeddings, llm=llm)
    with st.spinner('Processing...'):
        output, _ = query_service.process_query(prompt, memory)

    response_text = output.get("output_text", "Sorry, I couldn't process that.")
    messages.append({"role": "assistant", "content": response_text})
    
    with st.chat_message("assistant"):
        st.markdown(response_text)


def main():
    manage_users_ui()
    manage_conversations_ui()
    conversation_ui()

if __name__ == "__main__":
    main()
