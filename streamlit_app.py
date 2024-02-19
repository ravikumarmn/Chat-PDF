import streamlit as st
from service import QueryService, DocumentProcessingService
from user import User
from pathlib import Path
from utils import load_config
import os

config = load_config()

document_processing_service = DocumentProcessingService()
user_class = User()

if 'users' not in st.session_state:
    st.session_state.users = {}

def add_user_or_conversation(user_id=None):
    if user_id is None or user_id not in st.session_state.users:
        user_id, _ = user_class.create_new_user()
        st.session_state.users[user_id] = {'conversations': [], 'current_conversation': None}
    else:
        _, conversation_id = user_class.create_new_user()
        if conversation_id not in st.session_state.users[user_id]['conversations']:
            st.session_state.users[user_id]['conversations'].append(conversation_id)

with st.sidebar:
    st.header("Users and Conversations")
    selected_user_id = st.selectbox("Select User", options=list(st.session_state.users.keys()), index=0)
    if st.button("Add New User"):
        add_user_or_conversation()
    if st.button("Add New Conversation"):
        add_user_or_conversation(selected_user_id)

    if selected_user_id:
        selected_conversation_id = st.selectbox(
            "Select Conversation",
            options=st.session_state.users[selected_user_id]['conversations'],
            index=0 if st.session_state.users[selected_user_id]['conversations'] else -1
        )
        if selected_conversation_id:
            st.session_state.users[selected_user_id]['current_conversation'] = selected_conversation_id

if selected_user_id and st.session_state.users[selected_user_id]['current_conversation']:
    st.write(f"User: {selected_user_id}, Conversation: {st.session_state.users[selected_user_id]['current_conversation']}")

    uploaded_file = st.file_uploader("Choose a file to upload")
    if uploaded_file is not None:
        document_processing_service.process_uploaded_file(
            uploaded_file, selected_user_id, st.session_state.users[selected_user_id]['current_conversation']
        )
        st.success("File uploaded successfully.")

    query_text = st.text_input("Query Text", "")
    if st.button("Submit Query"):
        if query_text:
            db_path = Path(config["vector_store_dir"], selected_user_id, st.session_state.users[selected_user_id]['current_conversation'])
            query_service = QueryService(db_path=db_path)
            db_path = os.path.join("vectorstore",selected_user_id, st.session_state.users[selected_user_id]['current_conversation'])
            if os.path.exists(db_path):
                conversation_memory = document_processing_service.get_conversation_memory(
                    selected_user_id, st.session_state.users[selected_user_id]['current_conversation']
                )
            else:
                conversation_memory = document_processing_service.initialize_conversation_memory(
                    selected_user_id, st.session_state.users[selected_user_id]['current_conversation']
                )
                
            output, reranked_docs = query_service.process_query(
                query_text, conversation_memory
            )
            
            st.write("Response:", output.get("output_text", "No response text available."))
            
            # if reranked_docs:
            #     for idx, doc in enumerate(reranked_docs):
            #         st.write(f"Document {idx+1}:", doc.page_content)
            #         for key, value in doc.metadata.items():
            #             st.write(f"{key}: {value}")
        else:
            st.error("Query text is required.")
else:
    st.write("Please select a user and a conversation to start.")
