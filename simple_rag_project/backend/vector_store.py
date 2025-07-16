import os
from time import sleep
from typing import List

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DELAY = 0.02

class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(DELAY)
        return self.embedding.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        sleep(DELAY)
        return self.embedding.embed_query(text)
    

def create_vector_db(texts, collection_name="chroma"):
    embeddings = OpenAIEmbeddings()

    proxy_embedding = EmbeddingProxy(embeddings)

    db = Chroma(
        collection_name=collection_name,
        embedding_function=proxy_embedding,
        persist_directory=os.path.join("db/", collection_name)
    )

    db.add_documents(texts)

    return db

def get_vector_db(collection_name="chroma"):
    embeddings = OpenAIEmbeddings()

    proxy_embedding = EmbeddingProxy(embeddings)

    db = Chroma(
        collection_name=collection_name,
        embedding_function=proxy_embedding,
        persist_directory=os.path.join("db/", collection_name)
    )

    return db