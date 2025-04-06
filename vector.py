from langchain_ollama import OllamaEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import sqlite3
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./faiss_index"
sqlite_db_path = "restaurant_reviews_metadata.db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    # Initialize SQLite for metadata
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM metadata")  # clear old data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            rating REAL,
            date TEXT
        )
    """)
    
    
    for i, row in df.iterrows():
        doc_id = str(i)
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]}
        )
        documents.append(document)
        ids.append(doc_id)

        cursor.execute("INSERT INTO metadata (id, rating, date) VALUES (?, ?, ?)",
                       (doc_id, row["Rating"], row["Date"]))

    conn.commit()
    conn.close()

    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(db_location)

else:
    vector_store = FAISS.load_local(db_location, embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
