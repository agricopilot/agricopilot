# 🌱 AgriCopilot AI Backend

AgriCopilot AI Backend is a **FastAPI-powered service** that provides AI-driven tools for farmers, agricultural experts, and marketplaces. It leverages Meta LLaMA models and vector search to deliver intelligent and multilingual support for crop health, disaster management, and product recommendations.

---

## 🚀 Features
- **Crop Doctor** → Diagnose crop diseases and suggest treatments.
- **Multilingual Chat** → Communicate with farmers in their local language.
- **Disaster Summarizer** → Summarize disaster reports into actionable steps.
- **Marketplace Recommendation** → Suggest relevant matches for buying/selling farm products.
- **Vector Search** → Search relevant knowledge base content.

---

## 🛠️ Tech Stack
- **FastAPI** (Python framework for APIs)
- **LangChain + Meta LLaMA models** (AI model orchestration)
- **FAISS** (vector search for semantic retrieval)
- **Docker** (for containerized deployment)

---

## ⚙️ Setup & Installation
```bash
# Clone the repo
git clone https://github.com/<your-org>/agricopilot-backend.git
cd agricopilot-backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app:app --host 0.0.0.0 --port 7860
