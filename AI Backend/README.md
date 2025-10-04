# AgriCopilot AI Backend

AgriCopilot is an **AI-powered agricultural intelligence platform** developed by **Team Astra**.  
It combines **Conversational AI**, **Computer Vision**, and **Vector Intelligence** to empower farmers, researchers, and organizations with insights, crop diagnosis, and automated recommendations.

---

## ⚙️ TECH STACK
- **Framework:** FastAPI  
- **Language Models:** Meta LLaMA 3.1–8B Instruct  
- **Vision Model:** Meta LLaMA 3.2–11B Vision-Instruct  
- **Libraries:**
  - transformers  
  - torch  
  - accelerate  
  - sentencepiece  
  - langchain  
  - langchain_community  
  - langchain-huggingface  
  - faiss-cpu  
  - pandas  
  - Pillow  
  - datasets  
  - fastapi  
  - uvicorn

---

## 🗂 DIRECTORY STRUCTURE
📦 AgriCopilot  
 ┣ 📜 app.py               # Main FastAPI app (endpoints & model logic)  
 ┣ 📜 vector.py            # FAISS vector search and embedding logic  
 ┣ 📜 prepare_data.py      # Fetches and prepares datasets (PlantVillage, AfriQA, CrisisNLP)  
 ┣ 📜 requirements.txt     # Project dependencies  
 ┣ 📜 README.md            # Documentation  
 ┣ 📂 datasets/            # Downloaded datasets  
 ┗ 📂 faiss_index/         # FAISS embeddings index  

---

## 🔑 ENVIRONMENT VARIABLES
| Variable | Description | Example |
|-----------|-------------|----------|
| `PROJECT_API_KEY` | API authentication key | `agricopilot404` |
| `HUGGINGFACEHUB_API_TOKEN` | Token for gated HuggingFace models | `hf_XXXXXXXXXXXXXXXXXXXX` |

Set them before running:
export PROJECT_API_KEY="agricopilot404"
export HUGGINGFACEHUB_API_TOKEN="hf_your_access_token"

---

## ⚙️ INSTALLATION & SETUP
# Clone the repository
git clone https://github.com/agricopilot/agricopilot
cd AgriCopilot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

---

## 🚀 API ENDPOINTS

| Endpoint | Description | Model |
|-----------|--------------|--------|
| `POST /multilingual-chat` | General multilingual interaction | `meta-llama/Llama-3.1-8B-Instruct` |
| `POST /disaster-summarizer` | Summarizes crisis and disaster reports | `meta-llama/Llama-3.1-8B-Instruct` |
| `POST /marketplace` | Generates market and product recommendations | `meta-llama/Llama-3.1-8B-Instruct` |
| `POST /crop-doctor` | Diagnoses plant disease from image and text | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `POST /vector-search` | Performs semantic search using FAISS | `sentence-transformers/all-MiniLM-L6-v2` |

---

## 🧠 SAMPLE USAGE

### 🌍 Multilingual Chat
curl -X POST http://localhost:8000/multilingual-chat \
  -H "Authorization: Bearer agricopilot404" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I prevent leaf rust in maize?"}'

---

### 🌪 Disaster Summarizer
curl -X POST http://localhost:8000/disaster-summarizer \
  -H "Authorization: Bearer agricopilot404" \
  -H "Content-Type: application/json" \
  -d '{"report": "Severe flooding in Kano destroyed farmlands and displaced 1200 farmers."}'

---

### 🧑🏾‍🌾 Crop Doctor
curl -X POST "http://localhost:8000/crop-doctor" \
  -H "Authorization: Bearer agricopilot404" \
  -H "symptoms: Yellow leaves and black spots on tomato plants" \
  -F "image=@/path/to/tomato_leaf.jpg"

# Example Response
{
  "diagnosis": "The tomato leaves show signs of early blight. Apply a copper-based fungicide and rotate soil to reduce infection risk."
}

---

### 💹 Marketplace Insights
curl -X POST http://localhost:8000/marketplace \
  -H "Authorization: Bearer agricopilot404" \
  -H "Content-Type: application/json" \
  -d '{"product": "maize"}'

---

### 🧭 Vector Search
curl -X POST http://localhost:8000/vector-search \
  -H "Authorization: Bearer agricopilot404" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to improve soil fertility?"}'

---

## 🔐 AUTHENTICATION
All endpoints are secured with a Bearer Token:
Authorization: Bearer agricopilot404

---

## 🧰 ERROR HANDLING
{
  "error": "⚠️ Unexpected model error: ..."
}

---

## 📦 REQUIREMENTS
fastapi  
uvicorn  
transformers  
torch  
accelerate  
sentencepiece  
langchain  
langchain_community  
langchain-huggingface  
faiss-cpu  
pandas  
Pillow  
datasets  

---

## 🧩 VECTOR STORE LOGIC
Uses **LangChain + FAISS** for semantic search:  
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`  
- Automatically builds and caches the index from `/datasets`  
- Stores vectors locally in `/faiss_index`  

---

## 🧑🏾‍💻 DEVELOPMENT TEAM
**Team Astra**  
Innovating the Future of AI for Agriculture 🌱  
Email: agricopilot@gmail.com

---

## 🧭 LICENSE
Licensed under the **MIT License** — free for research, education, and innovation.
