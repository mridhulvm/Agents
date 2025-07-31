from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

db_location = "./chrome_restaurant_reviews_db"
is_db_exists = os.path.exists(db_location)

if not is_db_exists:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i),
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if not is_db_exists:
    vector_store.add_documents(documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
