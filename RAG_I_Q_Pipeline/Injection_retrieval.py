from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class Injection():
    def __init__(self, x, path_saved):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.loader = PyPDFLoader(x)
        self.chunker = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.path_saved = path_saved

    def RAG(self):
        docs = self.loader.load()
        chunks = self.chunker.split_documents(docs)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.path_saved
        )
        return vectordb

# Retrieval
class Retrieval():
    def __init__(self, path_saved):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.path_saved = path_saved

    def retrieve(self, query):
        vectordb = Chroma(
            persist_directory=self.path_saved,
            embedding_function=self.embeddings
        )
        docs = vectordb.similarity_search(query)
        return docs