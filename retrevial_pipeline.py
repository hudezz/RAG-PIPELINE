import warnings
# Squelch the messy deprecation warnings cleanly
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")

from langchain_chroma import Chroma
# SWITCH THIS IMPORT: Use the local HuggingFace module instead of OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

persistent_directory = "db/chroma_db"

print("=== Running Local Retrieval Pipeline ===")

# 1. Load the exact same local brain model we used to build the database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the existing local database folder on your Mac
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# 3. Define the query you want to test
query = "What kind of pushback, legal troubles, and public disapproval has Musk's car company faced?"
print(f"User Query: {query}")

# 4. Turn the database into a retriever component (fetch top 5 matches)
#retriever = db.as_retriever(search_kwargs={"k": 5})
retriever = db.as_retriever(
     search_type="similarity_score_threshold",
     search_kwargs={
      "k": 3,
         "score_threshold": 0.4 # Only return chunks with cosine similarity ≥ 0.3
     }
 )


# 5. Fetch the relevant text blocks!
print("Searching the database...")
relevant_docs = retriever.invoke(query)

print("\n--- Context Found ---")
if len(relevant_docs) == 0:
    print("No matching documents found in the database.")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i} (Source: {doc.metadata.get('source')}):")
        print(f"{doc.page_content}")
        print("-" * 50)