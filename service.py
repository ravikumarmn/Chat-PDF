import os
from pathlib import Path
from werkzeug.utils import secure_filename
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.python import PythonLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationSummaryBufferMemory
import shutil
from sentence_transformers import CrossEncoder
from langchain.output_parsers import pydantic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.llms.llamacpp import LlamaCpp
# from huggingface_hub import hf_hub_download
from langchain.llms.openai import OpenAI
from collections import defaultdict
from langchain.chains import ConversationChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.prompts import StringPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from huggingface_hub import hf_hub_download
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.memory import ConversationBufferMemory
from utils import load_config

config = load_config()


USE_OPENAI_MODEL = True

# TEMP_DIR.mkdir(parents=True, exist_ok=True)
# VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

def load_llama_2_llm():
    """Load the LlamaCpp model from Hugging Face Hub."""
    model_path = hf_hub_download(
        repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
        filename="llama-2-7b-chat.Q4_K_M.gguf",
        resume_download=True,
    )
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=100,  
        n_batch=1048,
        verbose=True,
        f16_kv=True,
        n_ctx=10000,
    )
    return llm

def load_llm():
    llm = OpenAI()
    return llm


if USE_OPENAI_MODEL:
    llm = load_llm()
else:
    llm = load_llama_2_llm()


class DirectoryLoader(BaseLoader):
    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.kwargs = kwargs

    def load(self):
        docs = list()
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = Path(os.path.join(root, file))
                if file_path.suffix == ".pdf":
                    loader = PyPDFLoader(file_path=str(file_path), **self.kwargs)
                elif file_path.suffix == ".txt":
                    loader = TextLoader(file_path=file_path, **self.kwargs)

                elif file_path.suffix == ".py":
                    loader = PythonLoader(file_path=file_path, **self.kwargs)
    
                elif file_path.suffix in [".md", ".markdown"]:
                    loader = UnstructuredMarkdownLoader(file_path= file_path, **self.kwargs)

                else:
                    print(f"Do not process the file: {file_path}")
                    continue
                
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
        return docs
    
class DocumentProcessingService:
    def __init__(self):
        self.user_conversation_memories = defaultdict(lambda: defaultdict(int))
        self.temp_directory = self.create_temp_directory()

    @staticmethod
    def create_temp_directory(temp_dir_name="tempdir", user_id = None, conversation_id = None):
        temp_directory = Path(temp_dir_name) / str(user_id) / str(conversation_id)

        temp_directory.mkdir(parents=True, exist_ok=True)
        return temp_directory

    def process_uploaded_file(self, file, user_id, conversation_id):
        temp_dir = self.create_temp_directory(user_id=user_id, conversation_id=conversation_id)
        # Use file.name to get the name of the uploaded file in Streamlit
        file_path = temp_dir / secure_filename(file.name)
        # In Streamlit, you should use file.getvalue() to read the file content
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        self.create_save_vector_db(user_id, conversation_id, temp_dir)
        self.delete_temp_directory()

    def initialize_conversation_memory(self, user_id, conversation_id):
        self.user_conversation_memories[user_id][conversation_id] =  ConversationBufferMemory(llm=OpenAI(), max_token_limit=config['max_token_limit'], memory_key="chat_history", input_key="human_input")

    def get_conversation_memory(self, user_id:str, conversation_id: str):
        if self.user_conversation_memories[user_id][conversation_id] == 0:
            self.initialize_conversation_memory(user_id, conversation_id)
        return self.user_conversation_memories[user_id][conversation_id]

    def create_save_vector_db(self, user_id, conversation_id, temp_dir):
        db_faiss_path = Path(config['vector_store_dir']) /  str(user_id) / str(conversation_id)
        db_faiss_path.parent.mkdir(parents=True, exist_ok=True)
        data_config = config['data_ingestion']

        loader = DirectoryLoader(data_dir=temp_dir)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=data_config['chunk_size'], chunk_overlap=data_config['chunk_overlap'])
        texts = text_splitter.split_documents(documents)
        
        # embeddings = HuggingFaceEmbeddings(model_name=data_config['embed_model'],
        #                                    model_kwargs={'device': config['device']})
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(db_faiss_path)

    def delete_temp_directory(self):
        """Deletes the temporary directory and all its contents."""
        if self.temp_directory.exists() and self.temp_directory.is_dir():
            shutil.rmtree(self.temp_directory)




class QueryService:
    def __init__(self, db_path):
        # embeddings = HuggingFaceEmbeddings(model_name=config['data_ingestion']['embed_model'],
        #                                    model_kwargs={'device': config['device']})
        embeddings = OpenAIEmbeddings()
        self.retriever = FAISS.load_local(str(db_path), embeddings=embeddings).as_retriever(k = 5)
        self.cross_encoder = CrossEncoder(config['data_ingestion']['cross_encoder_model'])
        self.llm = load_llm()

    def process_query(self, query_text, conversation_memory):
        queries = self.generate_multi_queries(query_text)
        docs = [self.retriever.invoke(query) for query in queries if query]
        unique_contents = set()
        unique_docs = []
        for sublist in docs:
            for doc in sublist:
                if doc.page_content not in unique_contents:
                    unique_docs.append(doc)
                    unique_contents.add(doc.page_content)
        unique_contents = list(unique_contents)

        reranked_docs = self.rerank_docs(unique_docs, query_text,top_k=2)  
        # template = """
        #         # AI Chatbot Response Template

        #         ## Task Overview
        #         As an AI chatbot, your primary objective is to understand and respond to user queries in an effective, accurate, and empathetic manner. Your response should be informed by the user's request and any context they have provided, offering helpful information, support, or guidance.

        #         ## User's Request
        #         - **Input Provided:** "{human_input}"
                
        #         ## Provided Context
        #         - **Contextual Details:** "{context}"

        #         {chat_history}

        #         ## Guidelines for Crafting a Response
        #         1. **Analyze the Request:** Carefully interpret the user's query. Identify the main points and any specific details that will influence your response.
                
        #         2. **Incorporate the Context:** Use the provided context to tailor your response. This may include addressing specific concerns, preferences, or scenarios mentioned by the user.
                
        #         3. **Formulate Your Response:** Your reply should be clear, informative, and directly address the user's query. Structure your response logically, including explanations, instructions, or recommendations as needed.
                
        #         4. **Maintain an Empathetic Tone:** Ensure your tone is understanding and supportive. It's important to make the user feel heard and assisted.
                
        #         5. **Ensure Accuracy:** Provide information that is reliable and verified. If uncertain, guide the user to resources where they can find additional help.
                
        #         6. **Clarity is Key:** Aim for a response that is straightforward and easy to understand, avoiding unnecessary complexity or technical jargon unless appropriate.

        #         ## Constructing the Response
        #         When constructing your response, begin with acknowledging the user's query, integrate the context to show understanding and personalization, then move on to providing a detailed and helpful answer. Conclude with an invitation for further questions or clarification.

        #         Remember, the essence of effective AI chatbot communication lies in being informative, empathetic, and engaging, ensuring that the user's needs are met with consideration and respect.
        #     """
        
        # PROMPT = CustomPromptTemplate(
        #                 input_variables=["history", "input"], template=template, context = " ".join([doc.page_content for doc in reranked_docs])
        #             )     
        # document_chain = create_stuff_documents_chain(self.llm, prompt)
    #     conversation_chain = ConversationChain(
    #         llm=self.llm(),
    #         prompt=prompt,
             
    # )
        # prompt = PromptTemplate(
        #         input_variables=["chat_history", "human_input", "context"], template=template
        #     )
        template = """You are a chatbot having a conversation with a human.

        Given the following extracted parts of a long document and a question, create a final answer.

        {context}

        {chat_history}
        Human: {human_input}
        Chatbot:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"], template=template
        )

        # prompt = PromptTemplate(
        #     input_variables=["chat_history", "human_input", "context"], template=template
        # )
        # memory_dict = conversation_memory

        chain = load_qa_chain(
            llm=OpenAI(temperature=0), 
            chain_type="stuff", 
            memory=conversation_memory,  # Pass the dictionary here
            prompt=prompt
        )

        output = chain({"input_documents": reranked_docs, "human_input": query_text})
        return output, reranked_docs
    
    def generate_multi_queries(self, query_text):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines. Only provide the query, no numbering.
        Original question: {question}"""
            )
        )
        queries = llm_chain.invoke({'question': query_text}).get("text")
        return [x.strip() for x in queries.split("\n") if x.strip()]
    
    def rerank_docs(self, docs, query_text, top_k=2):
        pairs = [{"pairs" : (query_text, doc.page_content), "metadata" : doc.metadata} for doc in docs]
        scores = self.cross_encoder.predict([pair['pairs'] for pair in pairs])
        scored_docs = sorted(zip(scores, pairs), key=lambda x: x[0], reverse=True)
        return [Document(page_content=doc['pairs'][1], metadata={**doc['metadata'], 'score': score}) for score, doc in scored_docs[:top_k]] # TODO: take the first and last score document.
