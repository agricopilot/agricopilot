# vector.py
import os
import glob
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------- CONFIG -----------------
VECTOR_PATH = "faiss_index"
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/app/huggingface_cache")

# Ensure cache dir exists
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# ----------------- EMBEDDINGS -----------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    cache_folder=HF_CACHE_DIR
)

# ----------------- VECTOR STORE -----------------
def build_vectorstore():
    """Build FAISS index from CSV datasets."""
    texts = []
    for file in glob.glob("datasets/*.csv"):
        try:
            df = pd.read_csv(file)
            if "text" in df.columns:
                texts.extend(df["text"].dropna().astype(str).tolist())
            else:
                # fallback: join all columns into one string
                for _, row in df.iterrows():
                    texts.append(" ".join(map(str, row.values)))
            print(f"✅ Loaded {len(df)} rows from {file}")
        except Exception as e:
            print(f"⚠️ Skipping {file}, error: {e}")

    if not texts:
        texts = ["AgriCopilot initialized knowledge base."]

    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(VECTOR_PATH)
    print("🎉 Vectorstore built with", len(texts), "documents")
    return vectorstore

def load_vector_store():
    """Load FAISS index if available, else build new one."""
    if os.path.exists(VECTOR_PATH):
        return FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        return build_vectorstore()

vectorstore = load_vector_store()

# ----------------- QUERY -----------------
def query_vector(query: str, k: int = 3):
    """Perform similarity search on FAISS index."""
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs]
