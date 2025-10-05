# vector.py
import os
import glob
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==============================
# CONFIGURATION
# ==============================
VECTOR_PATH = "faiss_index"
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/app/huggingface_cache")
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure cache directory exists
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# ==============================
# EMBEDDING SETUP
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    cache_folder=HF_CACHE_DIR,
    model_kwargs={"token": HF_TOKEN}
)

# ==============================
# VECTOR STORE OPERATIONS
# ==============================
def build_vectorstore():
    """Builds FAISS index from all CSV files in /datasets."""
    texts = []

    for file in glob.glob("datasets/*.csv"):
        try:
            df = pd.read_csv(file)

            if "text" in df.columns:
                # Primary text field
                texts.extend(df["text"].dropna().astype(str).tolist())
            else:
                # Combine all columns if no "text" column found
                texts.extend([" ".join(map(str, row.values)) for _, row in df.iterrows()])

            print(f"✅ Loaded {len(df)} rows from {file}")

        except Exception as e:
            print(f"⚠️ Skipping {file}, error: {e}")

    if not texts:
        texts = ["AgriCopilot initialized knowledge base."]

    print("📚 Building FAISS vector index...")
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(VECTOR_PATH)
    print(f"🎉 Vectorstore built successfully with {len(texts)} documents.")

    return vectorstore


def load_vector_store():
    """Loads FAISS index if it exists, otherwise builds a new one."""
    if os.path.exists(VECTOR_PATH):
        print("🔄 Loading existing FAISS index...")
        return FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🧩 No existing FAISS index found. Building a new one...")
        return build_vectorstore()


vectorstore = load_vector_store()

# ==============================
# VECTOR QUERY
# ==============================
def query_vector(query: str, k: int = 3):
    """
    Performs a semantic similarity search using FAISS.
    Returns a list of top-k relevant text chunks from the knowledge base.
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [d.page_content for d in docs]
    except Exception as e:
        print(f"⚠️ Vector query error: {e}")
        return ["No relevant knowledge found."]
