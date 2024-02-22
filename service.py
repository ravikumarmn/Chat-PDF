import os
import shutil

from pathlib import Path
from werkzeug.utils import secure_filename
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.python import PythonLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from sentence_transformers import CrossEncoder
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from collections import defaultdict
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from utils import load_config

config = load_config()

def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)



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
                    loader = UnstructuredMarkdownLoader(
                        file_path=file_path, **self.kwargs
                    )

                else:
                    print(f"Do not process the file: {file_path}")
                    continue

                loaded_docs = loader.load()
                docs.extend(loaded_docs)
        return docs


class DocumentProcessingService:
    def __init__(self):

        self.user_conversation_memories = defaultdict(recursive_defaultdict)
        self.temp_directory = self.create_temp_directory()

    @staticmethod
    def create_temp_directory(
        temp_dir_name="tempdir", user_id=None, conversation_id=None
    ):
        temp_directory = Path(temp_dir_name) / str(user_id) / str(conversation_id)

        temp_directory.mkdir(parents=True, exist_ok=True)
        return temp_directory

    def process_uploaded_file(self, file, user_id, conversation_id, embeddings):
        temp_dir = self.create_temp_directory(
            user_id=user_id, conversation_id=conversation_id
        )
        file_path = temp_dir / secure_filename(file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        self.create_save_vector_db(user_id, conversation_id, temp_dir,embeddings)
        self.delete_temp_directory()

    def initialize_conversation_memory(self, user_id, conversation_id, llm):
        self.user_conversation_memories[user_id][conversation_id] = ConversationBufferMemory(
                llm= llm,
                max_token_limit=config["max_token_limit"],
                memory_key="chat_history",
                input_key="human_input",
            )
        return self.user_conversation_memories[user_id][conversation_id]
    
    def get_conversation_memory(self, user_id: str, conversation_id: str, llm):
        if user_id not in self.user_conversation_memories:
            self.initialize_conversation_memory(user_id, conversation_id,llm)
            return self.user_conversation_memories[user_id][conversation_id]
        elif conversation_id not in self.user_conversation_memories[user_id]:
            self.initialize_conversation_memory(user_id, conversation_id,llm)
            return self.user_conversation_memories[user_id][conversation_id]
        else:
            return self.user_conversation_memories[user_id][conversation_id]
        

    def create_save_vector_db(self, user_id, conversation_id, temp_dir, embeddings):
        db_faiss_path = (
            Path(config["vector_store_dir"]) / str(user_id) / str(conversation_id)
        )
        db_faiss_path.parent.mkdir(parents=True, exist_ok=True)
        data_config = config["data_ingestion"]

        loader = DirectoryLoader(data_dir=temp_dir)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=data_config["chunk_size"],
            chunk_overlap=data_config["chunk_overlap"],
        )
        texts = text_splitter.split_documents(documents)

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(db_faiss_path)

    def delete_temp_directory(self):
        """Deletes the temporary directory and all its contents."""
        if self.temp_directory.exists() and self.temp_directory.is_dir():
            shutil.rmtree(self.temp_directory)


class QueryService:
    def __init__(self, db_path, embeddings, llm):
        self.llm = llm
        self.embeddings = embeddings
        self.db_path = db_path

        if db_path is not None:
            self.retriever = FAISS.load_local(
                str(db_path), embeddings=self.embeddings
            ).as_retriever(k=5)

        self.cross_encoder = CrossEncoder(
            config["data_ingestion"]["cross_encoder_model"]
        )

    def process_query(self, query_text, conversation_memory):
        if self.db_path is None:
            reranked_docs = ""
        else:
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

            reranked_docs = self.rerank_docs(unique_docs, query_text)

        # template = """You are a chatbot having a conversation with a human.

        # Given the following extracted parts of a long document and a question, create a final answer. 

        # {context}

        # {chat_history}
        # Human: {human_input}
        # Chatbot:"""
        template = """
            As a chatbot, engage in a dialogue with a user. Utilize the provided information to construct a succinct and relevant response.

            - Context: {context}

            - Previous Dialogue:
            {chat_history}

            - User's Query: {human_input}

            Respond concisely.
            - Chatbot's Reply:
        """

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"],
            template=template,
        )

        chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            memory=conversation_memory,  
            prompt=prompt,
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
        Original question: {question}""",
            ),
        )
        queries = llm_chain.invoke({"question": query_text}).get("text")
        return [x.strip() for x in queries.split("\n") if x.strip()]

    def rerank_docs(self, docs, query_text):
        pairs = [
            {"pairs": (query_text, doc.page_content), "metadata": doc.metadata}
            for doc in docs
        ]
        scores = self.cross_encoder.predict([pair["pairs"] for pair in pairs])
        scored_docs = sorted(zip(scores, pairs), key=lambda x: x[0], reverse=True)

        selected_docs = (
            [scored_docs[0], scored_docs[-1]] if len(scored_docs) > 1 else scored_docs
        )

        reranked_docs = [
            Document(
                page_content=doc[1]["pairs"][1],
                metadata={**doc[1]["metadata"], "score": doc[0]},
            )
            for doc in selected_docs
        ]

        return reranked_docs



# document_processing_service = DocumentProcessingService()
